import time

from jax import random, jit
from jax_md import space, energy, simulate


# configs
N = 32
dt = 1e-1
temperature = 0.1
box_size = 5.0
n_iter = 1000

# R: current position
# dR: displacement
# displacement(Ra, Rb):
#   dR = Ra - Rb
# periodic displacement(Ra, Rb):
#   dR = Ra - Rb
#   np.mod(dR + side * f32(0.5), side) - f32(0.5) * side
# periodic shift:
#   np.mod(R + dR, side)
# shift:
#   R + dR
displacement, shift = space.periodic(box_size)

# dr: pairwise distances
# epsilon: Interaction energy scale (const)
# alpha: interaction stiffness
# dr = distance(R)
# U(dr) = np.where(dr < 1.0, (1 - dr) ** 2, 0)
# energy_fn(R) = diagonal_mask(U(dr))
energy_fn = energy.soft_sphere_pair(displacement)

# simulation init
# force(energy) = -d(energy)/dR
# xi = random.normal(R.shape, R.dtype)
# gamma = 0.1
# nu = 1 / (mass * gamma)
# dR = force(R) * dt * nu + np.sqrt(2 * temperature * dt * nu) * xi
# BrownianState(position, mass, rng)
pos_key, sim_key = random.split(random.PRNGKey(0))
R = random.uniform(pos_key, (N, 2), maxval=box_size)
init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt, temperature)
apply_fn = jit(apply_fn)
state = init_fn(sim_key, R)

# start simulation
time_elapsed = time.perf_counter_ns()
for i in range(n_iter):
    if not i or (i + 1) % 100 == 0:
        print('Running iteration {} ...'.format(i + 1))
    state = apply_fn(state)
time_elapsed = time.perf_counter_ns() - time_elapsed
print(state.position)
print('Simulation finished in {:.3f}s, {:.3f}ms per iteration.'.format(time_elapsed / 1e9, time_elapsed / 1e6 / n_iter))
