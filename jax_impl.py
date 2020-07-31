import time

import jax.numpy as jnp
from jax import random, jit
from jax_md import space, energy, simulate


# configs
N = 32
dt = 1e-1
temperature = 0.1
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
displacement, shift = space.free()

# dr: pairwise distances
# epsilon: interaction energy scale (const)
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
R = random.uniform(pos_key, (N, 2), dtype=jnp.float32)
init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt, temperature)
apply_fn = jit(apply_fn)
state = init_fn(sim_key, R)

# start simulation
time_elapsed = time.perf_counter_ns()
for i in range(n_iter):
    state = apply_fn(state)
    if not i or (i + 1) % 100 == 0:
        total_time = time.perf_counter_ns() - time_elapsed
        print('Finish iteration {} ({:.3f}ms, {:.3f}ms per iteration)'
              .format(i + 1, total_time / 1e6, total_time / (1 if not i else 100) / 1e6))
        time_elapsed = time.perf_counter_ns()
print('Simulation done!')
