import numpy as np

import mindspore.ops.composite as C
from mindspore import ms_function
from mindspore import context
from mindspore import Tensor


@ms_function
def func(x, y):
    return -2 * x - y


context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
x = Tensor(np.ones((3, 3)))
y = Tensor(np.ones((3, 3)))
dx = C.grad(func)(x, y)
print(dx)
