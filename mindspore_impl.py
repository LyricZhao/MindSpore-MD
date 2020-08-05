import numpy as np
import math
import time
import os

import mindspore
import mindspore.ops.composite as C
from mindspore import ms_function
from mindspore import context
from mindspore import Tensor
import mindspore.ops.functional as F
import mindspore.ops.operations as P


# Disable MindSpore warning output
os.environ['GLOG_v'] = '3'


# MD configs
dims = 2
dt = 1e-1
temperature = 0.1
gamma = 0.1
mass = 1.0
nu = 1.0 / (mass * gamma)
xi_c = math.sqrt(2.0 * temperature * dt * nu)
dtype = mindspore.float32


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
    # TODO: plus 1e-6 is not accurate, a safe mask is better
    dr = F.sqrt(reduce_sum(dr * dr, -1) + 1.1920928955078125e-07)
    U = max_op(1 - dr, 0)
    U = reduce_sum(U * U * mask) * 0.5 * 0.5
    return U


@ms_function
def force_fn(R: Tensor, mask: Tensor):
    return -C.grad(energy_fn)(R, mask)


@ms_function
def apply_fn(R, mask, xi):
    dR = force_fn(R, mask) * dt * nu + xi_c * xi
    return shift(R, dR)


def run(N=32, n_iter=1000, with_graph_mode=True, save_ir_graph=False):
    # Runtime configs
    context.set_context(mode=(context.GRAPH_MODE if with_graph_mode else context.PYNATIVE_MODE),
                        device_target='GPU', save_graphs=save_ir_graph, save_graphs_path='.')

    # Simulation init
    R = Tensor(np.random.uniform(size=(N, dims)), dtype=dtype)
    mask = Tensor(1.0 - np.eye(N), dtype=dtype)

    # Start simulation
    times = []
    for i in range(n_iter):
        xi = Tensor(np.random.normal(size=R.shape), dtype=dtype)
        time_start = time.perf_counter_ns()
        R = apply_fn(R, mask, xi)
        time_end = time.perf_counter_ns()
        times.append(time_end - time_start)

    # Finish with profiling times
    return times


if __name__ == '__main__':
    print('Running MindSpore implement ... ', end='')
    time_elapsed = time.perf_counter_ns()
    run(32, 1000, True, True)
    time_elapsed = (time.perf_counter_ns() - time_elapsed) / 1e6
    print('done in {:.3f}ms!'.format(time_elapsed))
