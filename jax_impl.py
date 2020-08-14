import time
import os
import numpy as np

# Disable TensorFlow-XLA warning output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable TensorFlow-XLA memory pre-allocation
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Dump IR and graphs
# os.environ['XLA_FLAGS'] = "--xla_dump_to=."
# os.environ['TF_DUMP_GRAPH_PREFIX'] = '.'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_clustering_debug'


def run(N=32, n_iter=1000, with_jit=True):
    import jax.numpy as jnp
    from jax import random, jit
    from jax_md import space, energy, simulate

    # MD configs
    dt = 1e-1
    temperature = 0.1

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

    # Simulation init
    # dr: pairwise distances
    # epsilon: interaction energy scale (const)
    # alpha: interaction stiffness
    # dr = distance(R)
    # U(dr) = np.where(dr < 1.0, (1 - dr) ** 2, 0)
    # energy_fn(R) = diagonal_mask(U(dr))
    energy_fn = energy.soft_sphere_pair(displacement)

    # force(energy) = -d(energy)/dR
    # xi = random.normal(R.shape, R.dtype)
    # gamma = 0.1
    # nu = 1 / (mass * gamma)
    # dR = force(R) * dt * nu + np.sqrt(2 * temperature * dt * nu) * xi
    # BrownianState(position, mass, rng)
    pos_key, sim_key = random.split(random.PRNGKey(0))
    R = random.uniform(pos_key, (N, 2), dtype=jnp.float32)
    init_fn, apply_fn = simulate.brownian(energy_fn, shift, dt, temperature)
    if with_jit:
        apply_fn = jit(apply_fn)
    state = init_fn(sim_key, R)

    # Start simulation
    times = []
    for i in range(n_iter):
        time_start = time.perf_counter_ns()
        state = apply_fn(state)
        time_end = time.perf_counter_ns()
        times.append(time_end - time_start)

    # Finish with profiling times
    return times


if __name__ == '__main__':
    print('Running JAX with JIT implement ... ', end='')
    time_elapsed = time.perf_counter_ns()
    times = run(4096, 100, True)
    time_elapsed = (time.perf_counter_ns() - time_elapsed) / 1e6
    time_per_iter = np.mean(times[1:]) / 1e6
    print('done in {:.3f}ms ({:.3f}ms per iteration)!'.format(time_elapsed, time_per_iter))
