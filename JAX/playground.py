from jax import random
from jax_md import space, energy, simulate

# configs
N = 32
dt = 1e-1
temperature = 0.1
box_size = 5.0

# displacement:
#   np.mod(dR + side * f32(0.5), side) - f32(0.5) * side
# periodic shift:
#   np.mod(R + dR, side)
# shift:
#   R + dR
displacement, shift = space.periodic(box_size)

# TODO: figure it out
energy_fn = energy.soft_sphere_pair(displacement)

# simulate
pos_key, sim_key = random.split(random.PRNGKey(0))
pos = random.uniform(pos_key, (N, 2), maxval=box_size)
init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt, temperature)
state = init_fn(sim_key, pos)
for i in range(1000):
    if not i or (i + 1) % 100 == 0:
        print('Running iteration {} ...'.format(i + 1))
    state = apply_fn(state)
print(state.position)
