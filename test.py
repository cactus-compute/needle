import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Number of devices: {jax.device_count()}")
print(f"Available devices: {jax.devices()}")

# Run a simple calculation
x = jnp.ones((2, 2))
y = jnp.dot(x, x)
print(f"Calculation result: {y}")
