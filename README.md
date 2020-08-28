## MindSpore-MD

A simple example of Molecular Dynamics simulation on MindSpore and JAX-MD.

### Run Simulation

```shell script
python benchmark.py
```

Change the dict `configs` in `benchmark.py` to configure the simulation.

### Multi-GPU MindSpore Simulation

```shell script
mpirun -n 8 mindspore_multi_gpu.py
```

Change `N` and other variables in `__main__` function to configure the simulation.