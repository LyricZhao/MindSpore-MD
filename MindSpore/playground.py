import numpy as np
import math
import time

import mindspore
import mindspore.ops.composite as C
from mindspore import ms_function
from mindspore import context
from mindspore import Tensor
import mindspore.ops.functional as F
import mindspore.ops.operations as P

# MD configs
N = 32
dims = 2
dt = 1e-1
box_size = 5.0
temperature = 0.1
n_iter = 1000
gamma = 0.1
mass = 1.0
nu = 1.0 / (mass * gamma)
xi_c = math.sqrt(2.0 * temperature * dt * nu)
dtype = mindspore.float32


# Runtime configs
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

# Functions
reduce_sum = P.ReduceSum()
max_op = P.Maximum()


@ms_function
def displacement(Ra: Tensor, Rb: Tensor):
    dR = Ra - Rb
    # periodic
    # np.mod(dR + box_size * 0.5, box_size) - box_size * 0.5
    return dR


@ms_function
def pairwise_displacement(R: Tensor):
    dR = F.expand_dims(R, 1) - F.expand_dims(R, 0)
    # periodic
    # np.mod(dR + box_size * 0.5, box_size) - box_size * 0.5
    return dR


@ms_function
def shift(R: Tensor, dR: Tensor):
    # periodic
    # np.mod(R + dR, side)
    return R + dR


@ms_function
def energy_fn(R: Tensor, mask: Tensor):
    dr = pairwise_displacement(R)
    dr = F.sqrt(reduce_sum(dr * dr, -1))
    U = max_op(1 - dr, 0)
    U = U * U * mask * 0.5
    return U


@ms_function
def force_fn(R: Tensor, mask: Tensor):
    return -C.grad(energy_fn)(R, mask)


@ms_function
def apply_fn(R, mask, xi):
    dR = force_fn(R, mask) * dt * nu + xi_c * xi
    return shift(R, dR)


R = Tensor(np.random.normal(size=(N, dims)), dtype=dtype)
mask = Tensor(1.0 - np.eye(N), dtype=dtype)

# start simulation
time_elapsed = time.perf_counter_ns()
for i in range(n_iter):
    if not i or (i + 1) % 1 == 0:
        print('Running iteration {} ...'.format(i + 1))
    xi = Tensor(np.random.normal(size=R.shape), dtype=dtype)
    R = apply_fn(R, mask, xi)
time_elapsed = time.perf_counter_ns() - time_elapsed
print('Simulation finished in {:.3f}s, {:.3f}ms per iteration.'.format(time_elapsed / 1e9, time_elapsed / 1e6 / n_iter))
