from jax import random, jit
from jax_md import energy, space, simulate


N = 32
dt = 1e-1
temperature = 0.1
box_size = 5.0
key = random.PRNGKey(0)
displacement, shift = space.periodic(box_size)
energy_fn = energy.soft_sphere_pair(displacement)


def simulation(key):
    pos_key, sim_key = random.split(key)
    R = random.uniform(pos_key, (N, 2), maxval=box_size)
    init_fn, apply_fn = simulate.brownian(
        energy_fn, shift, dt, temperature)
    state = init_fn(sim_key, R)
    for i in range(1000):
        state = apply_fn(state)
    return state.position


positions = simulation(key)
print(positions)
