import math
import os
import time

import mindspore.nn as nn
import mindspore.ops.composite as C
import mindspore.ops.operations as P
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.train import Model

# N must be fixed
# Overflow when N=32768
N = 16384

# Runtime configs
init('nccl')
rank_id = get_rank()
context.set_context(mode=context.GRAPH_MODE, device_target='GPU', save_graphs=True, save_graphs_path='.')
context.set_auto_parallel_context(parallel_mode='auto_parallel')

# Disable MindSpore warning output
os.environ['GLOG_v'] = '3'


class Energy(nn.Cell):
    def __init__(self):
        super(Energy, self).__init__()
        self.reduce_sum = P.ReduceSum()
        self.relu = P.ReLU()
        self.expand = P.ExpandDims()
        self.sqrt = P.Sqrt()

    def construct(self, R):
        dr = self.expand(R, 2) - self.expand(R, 1)
        dr = self.sqrt(self.reduce_sum(dr * dr, -1) + 1.1920928955078125e-07)
        U = self.relu(1 - dr)
        # BUG: the line below should be `U = self.reduce_sum(U * U) * 0.5 * 0.5`
        U = self.reduce_sum(U * U)
        return U


class Apply(nn.Cell):
    def __init__(self):
        super(Apply, self).__init__()
        self.energy_fn = Energy()
        self.grad = C.GradOperation('grad')
        self.dt = 0.1
        self.temperature = 0.1
        self.gamma = 0.1
        self.mass = 1.0
        self.nu = 1.0 / (self.mass * self.gamma)
        self.xi_c = math.sqrt(2.0 * self.temperature * self.dt * self.nu)
        self.random = P.StandardNormal(seed=1, seed2=2)

    def construct(self, R, _):
        force = -self.grad(self.energy_fn)(R)
        xi = self.random((1, N, 2))
        dR = force * self.dt * self.nu + self.xi_c * xi
        return R + dR


def run(n_iter=1000):
    # Simulation init
    apply = Apply()
    model = Model(apply)
    R = apply.random((1, N, 2))

    # Start simulation
    times = []
    for i in range(n_iter):
        time_start = time.perf_counter_ns()
        R = model.predict(R, Tensor(0))
        time_end = time.perf_counter_ns()
        times.append(time_end - time_start)

    # Finish with profiling times
    return times


if __name__ == '__main__':
    if rank_id == 0:
        print('Running MindSpore implement ... ', end='')
    time_elapsed = time.perf_counter_ns()
    times = run(100)
    time_elapsed = (time.perf_counter_ns() - time_elapsed) / 1e6
    time_per_iter = np.mean(times[1:]) / 1e6
    if rank_id == 0:
        print('done in {:.3f}ms ({:.3f}ms per iteration)!'.format(time_elapsed, time_per_iter))
