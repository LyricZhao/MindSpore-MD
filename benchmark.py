import os
import time
import numpy as np

if __name__ == '__main__':
    # Disable TensorFlow-XLA warning output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Disable TensorFlow-XLA memory pre-allocation
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # Disable MindSpore warning output
    os.environ['GLOG_v'] = '3'

    # Imports implements
    import jax_impl
    import mindspore_impl

    # Configs
    n_iter = 1000
    cases = [32, 64, 128, 256, 512, 1024, 2048]
    configs = [
        ('JAX', jax_impl.run, {'with_jit': False}),
        ('JAX with JIT', jax_impl.run, {'with_jit': True}),
        ('MindSpore', mindspore_impl.run, {'with_graph_mode': True})]

    # Run
    csv_rows = []
    for N in cases:
        csv_row = [N]
        for name, func, config in configs:
            print('Running {} implement with N={} ... '.format(name, N), flush=True, end='')
            time_elapsed = time.perf_counter_ns()
            times = func(N, n_iter, **config)
            time_elapsed = (time.perf_counter_ns() - time_elapsed) / 1e6
            print('done!')
            time_per_iter = np.mean(times[1:]) / 1e6
            print(' > Time used: {:.3f}ms'.format(time_elapsed))
            print(' > Time per iteration: {:.3f}ms'.format(time_per_iter))
            csv_row.append(time_per_iter)
        csv_rows.append(csv_row)

    # Write into csv
    with open('profiling.csv', 'w') as file:
        file.write(','.join(['N'] + [config[0] for config in configs]) + '\n')
        file.write('\n'.join([','.join([str(row[0])] + ['{:.3f}'.format(num) for num in row[1:]]) for row in csv_rows]))