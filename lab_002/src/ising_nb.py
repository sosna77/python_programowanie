import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit, prange

@njit(cache=True)
def calculate_energy(grid, i, j, N, J, B):
    s = grid[i,j]
    s_sum = 0
    for dx in range(-1,2):
        for dy in range(-1, 2):
            s_sum += grid[(i+dx)%N, (j+dy)%N]
    s_sum -= s
    return 2*s*(s_sum*J + B)

@njit(parallel=True, cache=True)
def total_energy(grid, J, B, N):
    energy = 0.0
    for i in prange(N):
        for j in prange(N):
            s_curr = grid[i,j]
            energy += -J*s_curr*(grid[(i+1)%N, j] + grid[i, (j+1)%N]) - B*s_curr
    return energy

@njit(cache=True)
def magnet(N, grid):
    return 1/(N**2)*np.sum(grid)

@njit(cache=True)
def microstep(grid, N, beta, J, B):
    i, j = np.random.randint(0, N, size=2)
    dE = calculate_energy(grid, i, j, N, J, B)
    if dE<0: grid[i,j] = -grid[i,j]
    else:
        p = np.exp(-beta*dE)
        if p>np.random.uniform(): grid[i,j] = -grid[i,j]  
@njit(cache=True)
def macrostep(grid, N, beta, J, B):
    for _ in range(N**2):
        microstep(grid, N, beta, J, B)
    return grid, magnet(N, grid), total_energy(grid, J, B, N)


def visualize(history, magnetization, energy):
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

    
    
    # ax = axs[1]
    # ax.plot(ts, magnetization)

    # ax = axs[2]
    # ax.plot(ts, energy)

    plt.tight_layout()

    plt.show()

def main():
    start_time = time.perf_counter()
    # ==== INITIALIZATION =====
    parser = argparse.ArgumentParser(description='Ising model simulation')

    parser.add_argument('-N', '--size', type=int, default=100, help='lattice size (NxN) (default=100)')
    parser.add_argument('-J', '--exchange', type=float, default=1.0, help='exchange coefficient (default=1.0)')
    parser.add_argument('-b', '--beta', type=float, default=0.1, help='temperature parameter (default=0.1)')
    parser.add_argument('-B', '--field', type=float, default=1.0, help='external magnetic field (default=1.0)')
    parser.add_argument('-M', '--steps', type=int, default=100, help='number of macrosteps (default=100)')

    args = parser.parse_args()

    N = args.size
    J = args.exchange
    beta = args.beta
    B = args.field
    M = args.steps

    print(f'RUNNING SIMULATION FOR: N={N}, J={J}, beta={beta}, B={B}, M={M}')

    grid = np.random.choice(np.array([-1,1]), size=(N, N))
    history = np.empty((M,N,N), dtype=np.int64)
    magnetization = np.empty(M, dtype=np.float64)
    energy = np.empty(M, dtype=np.float64)

    history[0] = grid
    magnetization[0] = magnet(N, grid)
    energy[0] = total_energy(grid, J, B, N)
    for i in range(1, M):
        g, m, e = macrostep(grid, N, beta, J, B)
        history[i] = g
        magnetization[i] = m
        energy[i] = e
    sim_end = time.perf_counter()
    print(f'NUMBIFIED SIMMULATION LASTED: {sim_end - start_time:4f} s')
    visualize(history, magnetization, energy)



if __name__=='__main__':
    main()