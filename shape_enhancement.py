import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import context


class Cell(nn.Cell):
    def __init__(self, n):
        super(Cell, self).__init__()
        self.n = n
        self.random = P.StandardNormal()

    def construct(self, x):
        y = self.random(x.shape)
        return x + y


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE)
    cell = Cell(4096)
    x = cell.random((4096, ))
    print(cell(x))
