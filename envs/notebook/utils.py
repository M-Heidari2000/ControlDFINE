from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML


def display_frames_as_gif(frames):
    """Displays a list of frames as a gif, with controls."""
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    
    patch = ax.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        fig, animate, frames = len(frames), interval=50
    )
    video = anim.to_html5_video()
    html = HTML(video)
    display(html)
    plt.close()

