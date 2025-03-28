from jax import numpy as jnp
import numpy as np
import pytest
from scipy.signal import fftconvolve

from estimate import fftconvolve_chunked


@pytest.mark.parametrize("chunk_size", [100000, 2**12])
def test_fftconvolve_chunked(chunk_size: int):
    # Generate some test data
    x = np.random.rand(1000000)  # large signal
    h = np.random.rand(1000)  # kernel (or impulse response)

    # Compute the convolution in chunks
    y_chunked = fftconvolve_chunked(jnp.array(x), jnp.array(h), chunk_size)
    y_chunked = np.array(y_chunked)

    # For verification, you might compute the full convolution for a smaller signal:
    y_full = fftconvolve(x, h, mode="same")
    np.testing.assert_allclose(
        y_chunked, y_full
    )  # should be True (within numerical precision)
