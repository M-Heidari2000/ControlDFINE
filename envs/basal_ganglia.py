"""
Cortico-basal-ganglia-thalamic mean-field environment.

Spatially uniform van Albada / NFTsim formulation with a Muller-style DBS
pulse source phi_x(t) as a single external drive with fixed couplings to
STN, GPi/SNr, and GPe.

Action space
------------
3-D, normalised to [-1, 1].  Physical bounds are set by the constructor
arguments action_low and action_high (default: [0, 0, 5e-5], [60, 200, 5e-4]).

Observation
-----------
Flat 90-D vector: obs_samples_per_step (10) consecutive snapshots of all
9 population firing rates packed in order.

State  (info["state"])
----------------------
9-D vector q — instantaneous population firing rates at the end of the
current public step.

Reward
------
Negative variance of the STN ("zeta", index 8) firing-rate sub-trace
inside the current observation window:

    reward = -Var( obs.reshape(10, 9)[:, 8] )

High beta oscillation  →  high variance  →  large negative reward.
Suppressed oscillation →  low variance   →  reward near 0.

Episode end
-----------
terminated = False always.
truncated  = True after `horizon` steps.
"""

import math
from typing import Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from scipy.optimize import least_squares
from scipy.signal import cont2discrete


# ---------------------------------------------------------------------------
# Plant constants
# ---------------------------------------------------------------------------

POPULATION_ORDER = ("e", "i", "r", "s", "d1", "d2", "p1", "p2", "zeta")
POPULATION_INDEX = {name: idx for idx, name in enumerate(POPULATION_ORDER)}
DBS_TARGET_ORDER = ("zeta", "p1", "p2")
DBS_TARGET_INDEX = torch.tensor(
    [POPULATION_INDEX[name] for name in DBS_TARGET_ORDER], dtype=torch.long
)
DBS_INPUT_ORDER = ("pulse_amp", "pulse_freq", "pulse_width")

FIXED_POINT_GUESS = np.asarray(
    [12.0, 12.0, 28.0, 14.0, 7.4, 3.5, 69.0, 48.0, 28.0], dtype=np.float64
)
DBS_EDGE_TOL = 1e-12

_STN_IDX = POPULATION_INDEX["zeta"]   # 8




class _ConnectionSpec:
    __slots__ = ("name", "pre", "post", "nu_key", "tau_key", "harmonic")

    def __init__(self, name, pre, post, nu_key, tau_key, harmonic):
        self.name     = name
        self.pre      = pre
        self.post     = post
        self.nu_key   = nu_key
        self.tau_key  = tau_key
        self.harmonic = harmonic


_CONNECTION_SPECS = (
    _ConnectionSpec("e_to_e",       "e",    "e",    "nu_ee",     "tau_ee",     True),
    _ConnectionSpec("i_to_e",       "i",    "e",    "nu_ei",     "tau_ei",     False),
    _ConnectionSpec("s_to_e",       "s",    "e",    "nu_es",     "tau_es",     False),
    _ConnectionSpec("e_to_i",       "e",    "i",    "nu_ie",     "tau_ie",     True),
    _ConnectionSpec("i_to_i",       "i",    "i",    "nu_ii",     "tau_ii",     False),
    _ConnectionSpec("s_to_i",       "s",    "i",    "nu_is",     "tau_is",     False),
    _ConnectionSpec("e_to_r",       "e",    "r",    "nu_re",     "tau_re",     True),
    _ConnectionSpec("s_to_r",       "s",    "r",    "nu_rs",     "tau_rs",     False),
    _ConnectionSpec("p1_to_r",      "p1",   "r",    "nu_rp1",    "tau_rp1",    False),
    _ConnectionSpec("e_to_s",       "e",    "s",    "nu_se",     "tau_se",     True),
    _ConnectionSpec("r_to_s",       "r",    "s",    "nu_sr",     "tau_sr",     False),
    _ConnectionSpec("p1_to_s",      "p1",   "s",    "nu_sp1",    "tau_sp1",    False),
    _ConnectionSpec("n_to_s",       None,   "s",    "nu_sn",     "tau_sn",     False),
    _ConnectionSpec("e_to_d1",      "e",    "d1",   "nu_d1e",    "tau_d1e",    True),
    _ConnectionSpec("s_to_d1",      "s",    "d1",   "nu_d1s",    "tau_d1s",    False),
    _ConnectionSpec("d1_to_d1",     "d1",   "d1",   "nu_d1d1",   "tau_d1d1",   False),
    _ConnectionSpec("e_to_d2",      "e",    "d2",   "nu_d2e",    "tau_d2e",    True),
    _ConnectionSpec("s_to_d2",      "s",    "d2",   "nu_d2s",    "tau_d2s",    False),
    _ConnectionSpec("d2_to_d2",     "d2",   "d2",   "nu_d2d2",   "tau_d2d2",   False),
    _ConnectionSpec("d1_to_p1",     "d1",   "p1",   "nu_p1d1",   "tau_p1d1",   False),
    _ConnectionSpec("p2_to_p1",     "p2",   "p1",   "nu_p1p2",   "tau_p1p2",   False),
    _ConnectionSpec("zeta_to_p1",   "zeta", "p1",   "nu_p1zeta", "tau_p1zeta", False),
    _ConnectionSpec("d2_to_p2",     "d2",   "p2",   "nu_p2d2",   "tau_p2d2",   False),
    _ConnectionSpec("p2_to_p2",     "p2",   "p2",   "nu_p2p2",   "tau_p2p2",   False),
    _ConnectionSpec("zeta_to_p2",   "zeta", "p2",   "nu_p2zeta", "tau_p2zeta", False),
    _ConnectionSpec("e_to_zeta",    "e",    "zeta", "nu_zetae",  "tau_zetae",  True),
    _ConnectionSpec("p2_to_zeta",   "p2",   "zeta", "nu_zetap2", "tau_zetap2", False),
)

_BRAINSTEM_CONNECTION_INDEX = next(
    idx for idx, spec in enumerate(_CONNECTION_SPECS) if spec.name == "n_to_s"
)


# ---------------------------------------------------------------------------
# Plant helpers
# ---------------------------------------------------------------------------

def _discretize_second_order_system(first_order_decay, second_order_decay, dt):
    a_mat = np.asarray(
        [[0.0, 1.0],
         [-first_order_decay * second_order_decay,
          -(first_order_decay + second_order_decay)]],
        dtype=np.float64,
    )
    b_mat = np.asarray([[0.0], [first_order_decay * second_order_decay]], dtype=np.float64)
    c_mat = np.eye(2, dtype=np.float64)
    d_mat = np.zeros((2, 1), dtype=np.float64)
    ad, bd, _, _, _ = cont2discrete((a_mat, b_mat, c_mat, d_mat), dt, method="zoh")
    return torch.tensor(ad, dtype=torch.float32), torch.tensor(bd[:, 0], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Plant (private — instantiated by BasalGanglia)
# ---------------------------------------------------------------------------

class _BasalGangliaPlant:

    def __init__(self, **kwargs):
        q_init = kwargs.get("q_init")

        self.dt                    = float(kwargs["dt"])
        self.dt_internal           = float(kwargs["dt_internal"])
        self.obs_sample_dt         = float(kwargs["obs_sample_dt"])
        self.steps_per_public_step = int(round(self.dt / self.dt_internal))
        self.steps_per_obs_sample  = int(round(self.obs_sample_dt / self.dt_internal))

        if self.steps_per_public_step <= 0 or self.steps_per_obs_sample <= 0:
            raise ValueError("dt and obs_sample_dt must be positive multiples of dt_internal")
        if self.steps_per_public_step % self.steps_per_obs_sample != 0:
            raise ValueError("dt must be an integer multiple of obs_sample_dt")

        self.obs_samples_per_step = self.steps_per_public_step // self.steps_per_obs_sample
        self.dim_q = len(POPULATION_ORDER)
        self.dim_u = len(DBS_INPUT_ORDER)
        self.dim_y = self.obs_samples_per_step * self.dim_q

        self.alpha     = float(kwargs["alpha"])
        self.beta      = float(kwargs["beta"])
        self.gamma_e   = float(kwargs["gamma_e"])
        self.sigma     = float(kwargs["sigma"])
        self.phi_n     = float(kwargs["phi_n"])
        self.phi_n_std = float(kwargs["phi_n_std"])
        self.dbs_alpha = float(kwargs.get("dbs_alpha", self.alpha))
        self.dbs_beta  = float(kwargs.get("dbs_beta",  self.beta))
        self.dbs_nu    = torch.tensor(
            [kwargs["nu_zetax"], kwargs["nu_p1x"], kwargs["nu_p2x"]], dtype=torch.float32
        )

        self.dend_ad,    self.dend_bd    = _discretize_second_order_system(self.alpha,     self.beta,     self.dt_internal)
        self.dbs_ad,     self.dbs_bd     = _discretize_second_order_system(self.dbs_alpha, self.dbs_beta, self.dt_internal)
        self.harmonic_ad, self.harmonic_bd = _discretize_second_order_system(self.gamma_e, self.gamma_e,  self.dt_internal)

        self.qmax = torch.tensor(
            [kwargs["qmax_e"],  kwargs["qmax_i"],  kwargs["qmax_r"],  kwargs["qmax_s"],
             kwargs["qmax_d1"], kwargs["qmax_d2"], kwargs["qmax_p1"], kwargs["qmax_p2"],
             kwargs["qmax_zeta"]],
            dtype=torch.float32,
        )
        self.theta = torch.tensor(
            [kwargs["theta_e"],  kwargs["theta_i"],  kwargs["theta_r"],  kwargs["theta_s"],
             kwargs["theta_d1"], kwargs["theta_d2"], kwargs["theta_p1"], kwargs["theta_p2"],
             kwargs["theta_zeta"]],
            dtype=torch.float32,
        )

        params = dict(kwargs)
        params.setdefault("nu_ie",  params["nu_ee"])
        params.setdefault("nu_ii",  params["nu_ei"])
        params.setdefault("nu_is",  params["nu_es"])
        params.setdefault("tau_is", params["tau_es"])

        self.connection_specs             = _CONNECTION_SPECS
        self.connection_post_idx          = torch.tensor([POPULATION_INDEX[s.post] for s in self.connection_specs], dtype=torch.long)
        self.connection_pre_idx           = torch.tensor([POPULATION_INDEX[s.pre] if s.pre is not None else -1 for s in self.connection_specs], dtype=torch.long)
        self.connection_harmonic          = torch.tensor([s.harmonic for s in self.connection_specs], dtype=torch.bool)
        self.connection_map               = ~self.connection_harmonic
        self.connection_pre_idx_clamped   = torch.clamp(self.connection_pre_idx, min=0)
        self.connection_external_mask     = self.connection_pre_idx < 0
        self.connection_nu                = torch.tensor([params[s.nu_key]  for s in self.connection_specs], dtype=torch.float32)
        self.connection_delay_steps       = torch.tensor([int(round(params[s.tau_key] / self.dt_internal)) for s in self.connection_specs], dtype=torch.long)

        self.dbs_target_idx   = DBS_TARGET_INDEX
        self.max_delay_steps  = int(self.connection_delay_steps.max().item())
        self.delay_buffer_len = self.max_delay_steps + 1

        if q_init is not None:
            q_init_arr = np.asarray(q_init, dtype=np.float64)
            if q_init_arr.shape != (self.dim_q,):
                raise ValueError(f"q_init must have shape ({self.dim_q},)")
            self.q_init            = torch.tensor(q_init_arr, dtype=torch.float32)
            self.fixed_point_guess = q_init_arr.copy()
        else:
            self.q_init            = None
            self.fixed_point_guess = FIXED_POINT_GUESS.copy()

        # Runtime state — set by init_state()
        self.q = self.obs = self.y_seq = self.u_seq = None
        self.q_delay_buffer = self.q_delay_buffer_idx = None
        self.current_phi_n  = None
        self.phi = self.dphi = None
        self.v_dend = self.w_dend = None
        self.v_dbs  = self.w_dbs  = None
        self.dbs_phase = self.dbs_was_active = None
        self.t = None
        self.connection_post_idx_batched = self.soma_buffer = None

    # ------------------------------------------------------------------
    # Sigmoid
    # ------------------------------------------------------------------

    def _sigmoid(self, voltage):
        return self.qmax / (1.0 + torch.exp(-(voltage - self.theta) / self.sigma))

    # ------------------------------------------------------------------
    # Fixed point
    # ------------------------------------------------------------------

    def _fixed_point_residual(self, rates):
        q    = torch.tensor(rates, dtype=torch.float32)
        soma = torch.zeros(self.dim_q, dtype=torch.float32)
        for conn_idx, spec in enumerate(self.connection_specs):
            pre_rate = (torch.tensor(self.phi_n, dtype=torch.float32) if spec.pre is None
                        else q[POPULATION_INDEX[spec.pre]])
            soma[POPULATION_INDEX[spec.post]] += self.connection_nu[conn_idx] * pre_rate
        return (self._sigmoid(soma) - q).detach().cpu().numpy()

    def solve_fixed_point(self):
        lower  = np.full(self.dim_q, 1e-6, dtype=np.float64)
        upper  = self.qmax.detach().cpu().numpy().astype(np.float64) - 1e-6
        guess  = np.clip(self.fixed_point_guess, lower, upper)
        result = least_squares(
            self._fixed_point_residual, guess, bounds=(lower, upper),
            xtol=1e-12, ftol=1e-12, gtol=1e-12,
        )
        if not result.success:
            raise RuntimeError(f"Failed to solve basal ganglia fixed point: {result.message}")
        self.fixed_point_rates = torch.tensor(result.x, dtype=torch.float32)
        return self.fixed_point_rates.clone()

    # ------------------------------------------------------------------
    # Delay buffer
    # ------------------------------------------------------------------

    def _push_rates_to_delay_buffer(self, q_new):
        self.q_delay_buffer_idx = (self.q_delay_buffer_idx + 1) % self.delay_buffer_len
        self.q_delay_buffer[:, self.q_delay_buffer_idx, :] = q_new

    def _get_delayed_rates(self):
        delay_buffer_idx = (self.q_delay_buffer_idx - self.connection_delay_steps) % self.delay_buffer_len
        delayed_q_buffer = self.q_delay_buffer[:, delay_buffer_idx, :]
        gather_idx       = self.connection_pre_idx_clamped.view(1, -1, 1).expand(delayed_q_buffer.shape[0], -1, 1)
        delayed          = delayed_q_buffer.gather(2, gather_idx).squeeze(2)
        if bool(self.connection_external_mask.any()):
            delayed[:, self.connection_external_mask] = self.current_phi_n.unsqueeze(1)
        return delayed

    # ------------------------------------------------------------------
    # Second-order system helpers
    # ------------------------------------------------------------------

    def _apply_second_order_fixed_dt(self, v_val, w_val, drive, ad_mat, bd_vec):
        next_v = ad_mat[0, 0] * v_val + ad_mat[0, 1] * w_val + bd_vec[0] * drive
        next_w = ad_mat[1, 0] * v_val + ad_mat[1, 1] * w_val + bd_vec[1] * drive
        return next_v, next_w

    def _apply_second_order_exact_update(self, v_val, w_val, drive,
                                         first_order_decay, second_order_decay, dt_step):
        dt_tensor = torch.as_tensor(dt_step, dtype=v_val.dtype)
        while dt_tensor.ndim < drive.ndim:
            dt_tensor = dt_tensor.unsqueeze(-1)

        cascade_state = v_val + w_val / second_order_decay
        exp_first     = torch.exp(-first_order_decay * dt_tensor)
        next_cascade  = drive + (cascade_state - drive) * exp_first

        if math.isclose(first_order_decay, second_order_decay, rel_tol=0.0, abs_tol=1e-12):
            exp_second = exp_first
            next_v = (drive + (v_val - drive) * exp_second
                      + first_order_decay * (cascade_state - drive) * dt_tensor * exp_second)
        else:
            exp_second    = torch.exp(-second_order_decay * dt_tensor)
            coupling_term = (second_order_decay * (cascade_state - drive)
                             * (exp_first - exp_second) / (second_order_decay - first_order_decay))
            next_v = drive + (v_val - drive) * exp_second + coupling_term

        next_w = second_order_decay * (next_cascade - next_v)
        return next_v, next_w

    # ------------------------------------------------------------------
    # Propagators
    # ------------------------------------------------------------------

    def _update_harmonic_propagators(self, delayed_rates):
        if not bool(self.connection_harmonic.any()):
            return
        mask = self.connection_harmonic
        next_phi, next_dphi = self._apply_second_order_fixed_dt(
            self.phi[:, mask], self.dphi[:, mask], delayed_rates[:, mask],
            self.harmonic_ad, self.harmonic_bd,
        )
        self.phi[:, mask]  = next_phi
        self.dphi[:, mask] = next_dphi

    def _update_propagators(self):
        delayed_rates = self._get_delayed_rates()
        if bool(self.connection_map.any()):
            self.phi[:, self.connection_map]  = delayed_rates[:, self.connection_map]
            self.dphi[:, self.connection_map] = 0.0
        self._update_harmonic_propagators(delayed_rates)

    # ------------------------------------------------------------------
    # DBS state
    # ------------------------------------------------------------------

    def _sample_phi_n(self, batch_size):
        if self.phi_n_std <= 0.0:
            return torch.full((batch_size,), self.phi_n, dtype=torch.float32)
        return self.phi_n + self.phi_n_std * torch.randn(batch_size, dtype=torch.float32)

    def _parse_dbs_input(self, u_t):
        pulse_amp   = torch.clamp(u_t[:, 0], min=0.0)
        freqs       = torch.clamp(u_t[:, 1], min=0.0)
        widths      = torch.clamp(u_t[:, 2], min=0.0)
        safe_period = torch.where(freqs > 0.0, 1.0 / freqs, torch.full_like(freqs, float("inf")))
        widths      = torch.minimum(widths, safe_period)
        return pulse_amp, freqs, widths

    def _advance_dbs_state(self, v_dbs, w_dbs, dbs_phase, dbs_was_active,
                           pulse_amp, freqs, widths, return_mean_pulse=False):
        dbs_nu  = self.dbs_nu
        dbs_ad  = self.dbs_ad
        dbs_bd  = self.dbs_bd
        dt_step = self.dt_internal

        active    = (freqs > 0.0) & (widths > 0.0)
        periods   = torch.where(active, 1.0 / freqs, torch.ones_like(freqs))
        dbs_phase = torch.where(
            active & dbs_was_active,
            torch.remainder(dbs_phase, periods),
            torch.zeros_like(dbs_phase),
        )

        full_on    = active & (widths >= periods - DBS_EDGE_TOL)
        on_mask    = full_on | (active & (dbs_phase < widths))
        mean_pulse = torch.zeros_like(pulse_amp) if return_mean_pulse else None

        v_next = v_dbs.clone()
        w_next = w_dbs.clone()

        time_to_first_edge = torch.full_like(dbs_phase, float("inf"))
        nonfull_on  = active & (~full_on) &  on_mask
        nonfull_off = active & (~full_on) & ~on_mask
        time_to_first_edge = torch.where(nonfull_on,  widths  - dbs_phase, time_to_first_edge)
        time_to_first_edge = torch.where(nonfull_off, periods - dbs_phase, time_to_first_edge)

        has_first_edge = time_to_first_edge < (dt_step - DBS_EDGE_TOL)
        no_edge_mask   = ~has_first_edge
        next_phase     = torch.where(active, torch.remainder(dbs_phase + dt_step, periods), torch.zeros_like(dbs_phase))

        if not bool(has_first_edge.any()):
            pulse_rate = torch.where(on_mask, pulse_amp, torch.zeros_like(pulse_amp))
            dbs_drive  = pulse_rate.unsqueeze(1) * dbs_nu.unsqueeze(0)
            next_v, next_w = self._apply_second_order_fixed_dt(v_dbs, w_dbs, dbs_drive, dbs_ad, dbs_bd)
            mean_pulse = pulse_rate if return_mean_pulse else None
            return next_v, next_w, next_phase, active, mean_pulse

        if bool(no_edge_mask.any()):
            no_edge_pulse = torch.where(
                on_mask[no_edge_mask], pulse_amp[no_edge_mask], torch.zeros_like(pulse_amp[no_edge_mask])
            )
            no_edge_drive = no_edge_pulse.unsqueeze(1) * dbs_nu.unsqueeze(0)
            upd_v, upd_w  = self._apply_second_order_fixed_dt(
                v_next[no_edge_mask], w_next[no_edge_mask], no_edge_drive, dbs_ad, dbs_bd,
            )
            v_next[no_edge_mask] = upd_v
            w_next[no_edge_mask] = upd_w
            if mean_pulse is not None:
                mean_pulse[no_edge_mask] = no_edge_pulse

        if bool(has_first_edge.any()):
            edge_rows        = has_first_edge.nonzero(as_tuple=True)[0]
            first_widths     = widths[edge_rows]
            first_periods    = periods[edge_rows]
            first_amp        = pulse_amp[edge_rows]
            first_started_on = on_mask[edge_rows]
            first_dt         = time_to_first_edge[edge_rows]

            first_pulse = torch.where(first_started_on, first_amp, torch.zeros_like(first_amp))
            first_drive = first_pulse.unsqueeze(1) * dbs_nu.unsqueeze(0)
            first_v, first_w = self._apply_second_order_exact_update(
                v_next[edge_rows], w_next[edge_rows], first_drive,
                self.dbs_alpha, self.dbs_beta, first_dt,
            )
            v_next[edge_rows] = first_v
            w_next[edge_rows] = first_w
            if mean_pulse is not None:
                mean_pulse[edge_rows] = torch.where(
                    first_started_on,
                    first_amp * first_dt / dt_step,
                    torch.zeros_like(first_amp),
                )

            remaining_dt        = dt_step - first_dt
            second_started_on   = ~first_started_on
            time_to_second_edge = torch.where(first_started_on, first_periods - first_widths, first_widths)
            has_second_edge     = remaining_dt > (time_to_second_edge + DBS_EDGE_TOL)
            one_edge_mask       = ~has_second_edge

            one_edge_rows = edge_rows[one_edge_mask]
            if one_edge_rows.numel() > 0:
                second_pulse = torch.where(
                    second_started_on[one_edge_mask],
                    first_amp[one_edge_mask],
                    torch.zeros_like(first_amp[one_edge_mask]),
                )
                second_drive = second_pulse.unsqueeze(1) * dbs_nu.unsqueeze(0)
                second_v, second_w = self._apply_second_order_exact_update(
                    first_v[one_edge_mask], first_w[one_edge_mask], second_drive,
                    self.dbs_alpha, self.dbs_beta, remaining_dt[one_edge_mask],
                )
                v_next[one_edge_rows] = second_v
                w_next[one_edge_rows] = second_w
                if mean_pulse is not None:
                    mean_pulse[one_edge_rows] += second_pulse * remaining_dt[one_edge_mask] / dt_step

            multi_rows = edge_rows[has_second_edge]
            if multi_rows.numel() > 0:
                multi_started_on = second_started_on[has_second_edge]
                multi_amp        = first_amp[has_second_edge]
                multi_widths     = first_widths[has_second_edge]
                multi_periods    = first_periods[has_second_edge]
                multi_v          = first_v[has_second_edge].clone()
                multi_w          = first_w[has_second_edge].clone()
                multi_remaining  = remaining_dt[has_second_edge].clone()
                multi_edge_dt    = time_to_second_edge[has_second_edge].clone()

                while bool((multi_remaining > DBS_EDGE_TOL).any()):
                    segment_dt    = torch.minimum(multi_remaining, multi_edge_dt)
                    segment_pulse = torch.where(multi_started_on, multi_amp, torch.zeros_like(multi_amp))
                    segment_drive = segment_pulse.unsqueeze(1) * dbs_nu.unsqueeze(0)
                    multi_v, multi_w = self._apply_second_order_exact_update(
                        multi_v, multi_w, segment_drive,
                        self.dbs_alpha, self.dbs_beta, segment_dt,
                    )
                    if mean_pulse is not None:
                        mean_pulse[multi_rows] += torch.where(
                            multi_started_on,
                            multi_amp * segment_dt / dt_step,
                            torch.zeros_like(multi_amp),
                        )
                    multi_remaining  = multi_remaining - segment_dt
                    crossed_edge     = multi_remaining > DBS_EDGE_TOL
                    if not bool(crossed_edge.any()):
                        break
                    multi_started_on = torch.where(crossed_edge, ~multi_started_on, multi_started_on)
                    multi_edge_dt    = torch.where(multi_started_on, multi_widths, multi_periods - multi_widths)

                v_next[multi_rows] = multi_v
                w_next[multi_rows] = multi_w

        return v_next, w_next, next_phase, active, mean_pulse

    # ------------------------------------------------------------------
    # Public step
    # ------------------------------------------------------------------

    def _integrate_public_step(self, u_t):
        batch_size  = u_t.shape[0]
        sample_list = []
        pulse_amp, freqs, widths = self._parse_dbs_input(u_t)
        u_applied = torch.stack([pulse_amp, freqs, widths], dim=-1)

        for internal_idx in range(self.steps_per_public_step):
            self.current_phi_n = self._sample_phi_n(batch_size)
            self.phi[:, _BRAINSTEM_CONNECTION_INDEX] = self.current_phi_n

            intrinsic_drive = self.phi * self.connection_nu.unsqueeze(0)
            self.v_dend, self.w_dend = self._apply_second_order_fixed_dt(
                self.v_dend, self.w_dend, intrinsic_drive, self.dend_ad, self.dend_bd,
            )

            self.v_dbs, self.w_dbs, self.dbs_phase, self.dbs_was_active, _ = (
                self._advance_dbs_state(
                    self.v_dbs, self.w_dbs, self.dbs_phase, self.dbs_was_active,
                    pulse_amp, freqs, widths, return_mean_pulse=False,
                )
            )

            soma = self.soma_buffer.zero_()
            soma.scatter_add_(1, self.connection_post_idx_batched, self.v_dend)
            soma[:, self.dbs_target_idx] += self.v_dbs

            q_new  = self._sigmoid(soma)
            self.q = q_new
            self._push_rates_to_delay_buffer(q_new)
            self._update_propagators()

            if (internal_idx + 1) % self.steps_per_obs_sample == 0:
                sample_list.append(q_new.clone())

        obs_window = torch.stack(sample_list, dim=1)
        return q_new, obs_window.reshape(batch_size, -1), u_applied

    def init_state(self, num_seqs, num_steps):
        q0_init = self.q_init if self.q_init is not None else self.solve_fixed_point()
        q0      = q0_init.unsqueeze(0).repeat(num_seqs, 1)

        self.t     = -1
        self.u_seq = torch.zeros(num_seqs, num_steps, self.dim_u,  dtype=torch.float32)
        self.y_seq = torch.full((num_seqs, num_steps, self.dim_y), torch.nan, dtype=torch.float32)

        self.connection_post_idx_batched = self.connection_post_idx.unsqueeze(0).expand(num_seqs, -1)
        self.soma_buffer = torch.empty(num_seqs, self.dim_q, dtype=torch.float32)

        self.q             = q0.clone()
        self.obs           = q0.unsqueeze(1).repeat(1, self.obs_samples_per_step, 1).reshape(num_seqs, -1)
        self.current_phi_n = torch.full((num_seqs,), self.phi_n, dtype=torch.float32)

        self.q_delay_buffer     = q0.unsqueeze(1).repeat(1, self.delay_buffer_len, 1)
        self.q_delay_buffer_idx = self.delay_buffer_len - 1

        delayed_rates = self._get_delayed_rates()
        self.phi      = delayed_rates.clone()
        self.dphi     = torch.zeros(num_seqs, len(self.connection_specs), dtype=torch.float32)
        self.v_dend   = self.connection_nu.unsqueeze(0).repeat(num_seqs, 1) * self.phi
        self.w_dend   = torch.zeros_like(self.v_dend)
        self.v_dbs    = torch.zeros(num_seqs, len(DBS_TARGET_ORDER), dtype=torch.float32)
        self.w_dbs    = torch.zeros_like(self.v_dbs)
        self.dbs_phase      = torch.zeros(num_seqs, dtype=torch.float32)
        self.dbs_was_active = torch.zeros(num_seqs, dtype=torch.bool)

        # log t = 0 (initial observation)
        self.t += 1
        self.y_seq[:, self.t, :] = self.obs

    def step(self, u):
        u = u.float()
        if self.t >= self.y_seq.shape[1] - 1:
            raise IndexError("Basal ganglia plant stepped beyond the allocated horizon")
        self.q, self.obs, u_applied = self._integrate_public_step(u)
        self.u_seq[:, self.t, :] = u_applied
        self.t += 1
        self.y_seq[:, self.t, :] = self.obs


# ---------------------------------------------------------------------------
# Gym environment
# ---------------------------------------------------------------------------

class BasalGanglia(gym.Env):
    """
    Gym environment for beta-oscillation suppression via DBS.

    Parameters
    ----------
    action_low : array-like of shape (3,)
        Physical lower bounds for (pulse_amp, pulse_freq, pulse_width).
        Defaults to [0.0, 0.0, 5e-5].
    action_high : array-like of shape (3,)
        Physical upper bounds for (pulse_amp, pulse_freq, pulse_width).
        Defaults to [60.0, 200.0, 5e-4].
    horizon : int
        Maximum number of public steps per episode before truncation.
    **plant_kwargs
        Forwarded directly to the internal plant.  See the YAML configs
        (basal_ganglia_healthy.yaml / basal_ganglia_pd.yaml) for the full
        list of parameters (dt, dt_internal, obs_sample_dt, alpha, beta, ...).
    """

    x_dim = 9   # dim_q: one firing rate per neural population
    u_dim = 3   # (pulse_amp, pulse_freq, pulse_width)
    # y_dim is set in __init__ from plant.dim_y (depends on dt / obs_sample_dt)

    def __init__(
        self,
        action_low:  Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
        horizon: int = 1000,
        **plant_kwargs,
    ):
        super().__init__()

        self.action_low  = np.array(action_low  if action_low  is not None else [0.0,  0.0,   5e-5], dtype=np.float32)
        self.action_high = np.array(action_high if action_high is not None else [60.0, 200.0, 5e-4], dtype=np.float32)

        assert self.action_low.shape == (self.u_dim,), f"action_low must have shape ({self.u_dim},)"
        assert self.action_high.shape == (self.u_dim,), f"action_high must have shape ({self.u_dim},)"
        assert np.all(self.action_low < self.action_high), "action_low must be strictly less than action_high"

        self.plant   = _BasalGangliaPlant(**plant_kwargs)
        self.horizon = horizon
        self.y_dim   = self.plant.dim_y                           # 90 with default config
        self.obs_samples_per_step = self.plant.obs_samples_per_step   # 10

        self.action_space = spaces.Box(
            low   = -1.0,
            high  =  1.0,
            shape = (self.u_dim,),
            dtype = np.float32,
        )

        # State space: firing rates lie in [0, qmax] per population.
        self.state_space = spaces.Box(
            low   = np.zeros(self.x_dim, dtype=np.float32),
            high  = self.plant.qmax.numpy().copy(),
            shape = (self.x_dim,),
            dtype = np.float32,
        )

        # Observations: concatenated non-negative firing-rate windows.
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = np.inf,
            shape = (self.y_dim,),
            dtype = np.float32,
        )

        self._step = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalised [-1, 1]^3 to physical DBS parameters."""
        a = np.clip(action, -1.0, 1.0).astype(np.float32)
        return self.action_low + (a + 1.0) / 2.0 * (self.action_high - self.action_low)

    def _compute_reward(self, obs: np.ndarray) -> float:
        """Negative variance of the STN firing-rate sub-trace in this obs window."""
        stn_trace = obs.reshape(self.obs_samples_per_step, self.x_dim)[:, _STN_IDX]
        return -float(np.var(stn_trace))

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # horizon + 1: init_state logs step 0, then we take exactly `horizon`
        # steps reaching index horizon before the plant's internal guard fires.
        self.plant.init_state(num_seqs=1, num_steps=self.horizon + 1)

        # .numpy().copy() is required: plant tensors are modified in-place on
        # every subsequent step() call, so we must own our own arrays.
        obs   = self.plant.obs.squeeze(0).numpy().copy()  # (y_dim,)
        state = self.plant.q.squeeze(0).numpy().copy()    # (x_dim,)

        self._step = 0
        return obs, {"state": state}

    def step(self, action: np.ndarray):
        u_physical = self._denormalize_action(action)
        u_tensor   = torch.as_tensor(u_physical).unsqueeze(0)  # (1, 3) — batch dim

        self.plant.step(u_tensor)

        obs   = self.plant.obs.squeeze(0).numpy().copy()  # (y_dim,)
        state = self.plant.q.squeeze(0).numpy().copy()    # (x_dim,)

        reward     = self._compute_reward(obs)
        self._step += 1
        terminated = False
        truncated  = bool(self._step >= self.horizon)

        return obs, reward, terminated, truncated, {"state": state}
