from jax import numpy as jnp
import librosa
import numpy as np

from estimate import stft_in_batches


def test_stft_in_batches():
    """
    Test the stft_in_batches function against librosa's built-in stft for a random signal to ensure they match closely.
    """
    np.random.seed(0)
    y = np.random.randn(1, 1, 44100 * 100)  # ~1 second at 44.1 kHz

    n_fft = 2048
    hop_length = 512

    # Reference STFT using librosa (center=True)
    stft_ref = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)

    # Chunked STFT
    stft_chunked = stft_in_batches(
        jnp.array(y), n_fft=n_fft, hop_length=hop_length, chunk_size=1000
    )
    stft_chunked = np.array(stft_chunked)

    np.testing.assert_allclose(stft_ref, stft_chunked, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    test_stft_in_batches()
