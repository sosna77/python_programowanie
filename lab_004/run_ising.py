import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from ising import macrostep, total_energy, magnet, create_animation, save_magnetization_data, save_animation



def main():
    start_time = time.perf_counter()
    # ==== INITIALIZATION =====
    parser = argparse.ArgumentParser(description='Ising model simulation')

    parser.add_argument('-N', '--N', '--size', type=int, default=100, help='lattice size (NxN) (default=100)')
    parser.add_argument('-J','--J', '--exchange', type=float, default=1.0, help='exchange coefficient (default=1.0)')
    parser.add_argument('-b', '--beta', type=float, default=0.4406, help='temperature parameter (default=0.4406)')
    parser.add_argument('-B', '--B', '--field', type=float, default=0.0, help='external magnetic field (default=0.0)')
    parser.add_argument('-M', '--M', '--steps', type=int, default=1000, help='number of macrosteps (default=1000)')
    parser.add_argument('--show-animation', action='store_true', help='show animation after simulation')
    parser.add_argument('--magnetization-file', type=str, default=None, help='file to save magnetization')
    parser.add_argument('--animation-file', type=str, default=None, help='file to save animation')
    args = parser.parse_args()

    try:
        if args.size <= 0:
            raise ValueError('Grid size N must be greater than 0.')
        if args.beta < 0:
            raise ValueError('Beta must be positive.')
        if args.steps <= 0:
            raise ValueError('Number of steps must be greater than 0.')

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

        if args.show_animation or args.animation_file:
            anim, fig = create_animation(history, magnetization, energy)
            if args.animation_file:
                save_animation(anim, args.animation_file)
            if args.show_animation:
                plt.show()
            else:
                plt.close(fig)

        if args.magnetization_file:
            save_magnetization_data(args.magnetization_file, magnetization)

    except ValueError as e:
        print(f'Value error: {e}', file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f'File read/write error: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Unexpected error: {e}', file=sys.stderr)
        sys.exit(1)

if __name__=='__main__':
    main()