import numpy as np

from mindspore import ms_function
from mindspore import context
from mindspore import Tensor
import mindspore.ops.composite as C
import mindspore.ops.functional as F
import mindspore.ops.operations as P
import mindspore

# Runtime configs
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

# Functions
reduce_sum = P.ReduceSum()


@ms_function
def f_1_can_not_run(x):
    y = reduce_sum(x * x, -1)
    y = F.sqrt(y)
    return y


@ms_function
def f_2_can_run(x):
    y = reduce_sum(x * x, -1)
    return y


@ms_function
def f_3_can_run(x):
    y = F.sqrt(x)
    return y


x = Tensor(np.random.normal(size=(32, 2)), dtype=mindspore.float32)
print(C.grad(f_1_can_not_run)(x))  # compile error

x = Tensor(np.random.normal(size=(32, 2)), dtype=mindspore.float32)
print(C.grad(f_2_can_run)(x))

x = Tensor(np.random.normal(size=(32,)), dtype=mindspore.float32)
print(C.grad(f_3_can_run)(x))
