import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_animation(history, magnetization, energy):
    fig, axs = plt.subplots(1, 3, figsize=(12, 7))
    axs.flatten()
    ax = axs[0]

    im = ax.imshow(history[0], cmap='gray', vmin=-1, vmax=1)
    title_txt = ax.text(0.5, 1.05, 'Frame: 0',transform=ax.transAxes, ha='center', fontsize=16)
    ts = np.arange(len(history))

    mag, = axs[1].plot([],[], color='C1')
    en, = axs[2].plot([],[], color='C2')
    axs[1].set_xlim(0, len(history))
    axs[1].set_ylim(min(magnetization), max(magnetization))
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('m')
    axs[1].set_title('magnetisation')
    axs[2].set_xlim(0, len(history))
    axs[2].set_ylim(min(energy), max(energy))
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('H')
    axs[2].set_title('total energy')

    def update(i):
        im.set_data(history[i])
        title_txt.set_text(f'Frame: {i}')
        mag.set_data(ts[:i+1], magnetization[:i+1])
        en.set_data(ts[:i+1], energy[:i+1])

        return [im, title_txt, mag, en]
    
    anim = FuncAnimation(fig, update, frames=len(history), interval=20, blit=False)
    
    return anim, fig