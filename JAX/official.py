import jax.numpy as jnp
from jax import grad, pmap


def pairwise_displacement(R):
    dR = jnp.expand_dims(R, 1) - jnp.expand_dims(R, 0)
    return dR


def test_fn(R):
    dr = pairwise_displacement(R)
    dr = jnp.sqrt(jnp.sum(dr * dr, -1))
    return dr


R = jnp.array([[1.0, 0.98], [1.02, 1.01]], dtype=jnp.float32)
print(test_fn(R))
# print(grad(test_fn)(R))
