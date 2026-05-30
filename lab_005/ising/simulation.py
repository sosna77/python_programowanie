import numpy as np
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