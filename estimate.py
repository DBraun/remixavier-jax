"""
This is a JAX port of Colin Raffel and Dan Ellis's Remixavier:
Functions for aligning and approximately removing channel distortion in signals with related content.
"""

from functools import partial
import math
from typing import Tuple

import jax

jax.config.update("jax_enable_x64", True)
from jax import lax
from jax import numpy as jnp
from jax.scipy.optimize import minimize
from jax.scipy.signal import fftconvolve
from jax.experimental import checkify
import argbind
import audiotree.resample
import librosa
from librosax import stft, istft, amplitude_to_db
import numpy as np
import tqdm


def resample(x: jnp.ndarray, old_sr: int, new_sr: int):
    # add batch dimension
    x = jnp.expand_dims(x, axis=1)
    x = audiotree.resample.resample(x, old_sr, new_sr)
    x = jnp.squeeze(x, axis=1)
    return x


@jax.jit
def _fftconvolve(in1: jnp.ndarray, in2: jnp.ndarray):
    return fftconvolve(in1, in2, mode="full")


def fftconvolve_chunked(in1: jnp.ndarray, in2: jnp.ndarray, chunk_size: int):
    """
    Compute the FFT-based convolution of in1 and in2 in chunks using overlap-add,
    returning the same output as ``scipy.signal.fftconvolve(in1, in2, mode="same")``.

    Args:
        in1 (jnp.ndarray) : The first input signal (assumed to be very large).
        in2 (jnp.ndarray) : The second input signal (typically smaller than ``in1``).
        chunk_size (int): The size of each chunk for processing ``in1``.

    Returns:
        jnp.ndarray: The convolution result in "same" mode, i.e., the same length as ``in1``.
    """
    n1 = in1.shape[0]
    n2 = in2.shape[0]
    full_length = n1 + n2 - 1

    # Allocate an array for the full convolution result
    full_conv = jnp.zeros(full_length, dtype=jnp.result_type(in1, in2))

    # Process in1 in chunks using the overlap-add method
    for start in range(0, n1, chunk_size):
        # Extract the current chunk from in1
        chunk = in1[start : start + chunk_size]
        # Convolve the current chunk with in2 in 'full' mode
        conv_chunk = _fftconvolve(chunk, in2)
        # Add the convolution result into the correct position in the full_conv array
        full_conv = full_conv.at[start : start + conv_chunk.shape[0]].add(conv_chunk)

    # To mimic mode='same', we need to extract the central part of full_conv.
    # The offset should be (n2 - 1) // 2.
    offset = (n2 - 1) // 2
    result_same = full_conv[offset : offset + n1]
    return result_same


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _stft(
    y: jnp.ndarray,
    n_fft: int,
    hop_length: int = None,
    win_length: int = None,
    window: str = "hann",
):
    return stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
    )


def stft_in_batches(
    y: jnp.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = None,
    window: str = "hann",
    chunk_size: int = 2**13,
    dtype=jnp.complex128,
    verbose: bool = True,
):
    """
    Compute librosa-style STFT (center=True, zero-padded) in batches to reduce GPU memory usage, while allowing custom
    win_length.

    Args:
        y (np.ndarray): [shape=(..., T,)], real-valued time-domain signal.
        n_fft (int): Number of FFT components (window size).
        hop_length (int): Number of samples between successive frames.
        win_length (int | None): Each frame of audio is windowed by ``window`` of length ``win_length``
            and then padded with zeros to match ``n_fft``. If None, then
            `win_length = n_fft`.
        window (Union[str, tuple, number, function, np.ndarray]): Window specification, just like librosa. E.g., "hann",
            "hamming", etc.
        chunk_size (int): Number of frames to process per batch.
        dtype (jnp.dtype): Data type for the output STFT array.
        verbose (bool): Whether to print progress with ``tqdm``.

    Returns:
        jnp.ndarray: stft_out [shape=(1 + n_fft//2, n_frames)], complex
            STFT matrix of the entire signal, matching what you'd get
            from ``librosa.stft(...)`` with the same parameters (except center=True).
    """
    # Match librosa's frame calculation for center=True:
    T = y.shape[-1]
    n_frames = 1 + ((T + (n_fft // 2) * 2 - n_fft) // hop_length)

    # If win_length is None, match librosa default (win_length = n_fft)
    if win_length is None:
        win_length = n_fft

    # Allocate output array: shape = (1 + n_fft//2, n_frames)
    out_shape = y.shape[:-1] + (1 + n_fft // 2, n_frames)
    stft_out = jnp.empty(shape=out_shape, dtype=dtype)

    # Process frames in chunks of `chunk_size`
    for frame_start in tqdm.trange(
        0, n_frames, chunk_size, desc="STFT Batches", disable=not verbose
    ):
        frame_end = min(frame_start + chunk_size, n_frames)

        # We'll compute frames [frame_start, frame_end-1].
        # For center=True, frame i uses samples:
        #   [i * hop_length - n_fft//2,  i * hop_length + n_fft//2]
        start_sample = frame_start * hop_length - (n_fft // 2)
        end_sample = (
            (frame_end - 1) * hop_length + (n_fft // 2) + 1
        )  # +1 for Python slice end

        # Determine how much we need to pad on the left or right
        pad_left = max(0, -start_sample)
        pad_right = max(0, end_sample - T)

        # Slice the valid portion from y
        actual_start = max(start_sample, 0)
        actual_end = min(end_sample, T)
        y_sub = y[..., actual_start:actual_end]

        # Zero-pad if needed
        if pad_left or pad_right:
            pad_width = ((0, 0),) * (y_sub.ndim - 1) + ((pad_left, pad_right),)
            y_sub = jnp.pad(y_sub, pad_width=pad_width)

        # Compute STFT on this chunk with center=False (manual center logic above)
        stft_sub = _stft(
            y_sub,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )

        # stft_sub has shape (..., 1 + n_fft//2, frame_end - frame_start)
        if stft_sub.shape[-1] != (frame_end - frame_start):
            raise RuntimeError(
                f"Chunked STFT mismatch: got {stft_sub.shape[-1]} frames, "
                f"expected {frame_end - frame_start}."
            )

        # Place it in the output
        stft_out = stft_out.at[..., frame_start:frame_end].set(stft_sub)

    return stft_out


@checkify.checkify
@argbind.bind()
def align_over_window(
    a: jnp.ndarray,
    b: jnp.ndarray,
    max_offset: int,
    correlation_size: int,
    a_center: int = None,
    b_center: int = None,
    chunk_size: int = 2**14,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Correlates two signals over a subset of their samples

    Args:
        a (jnp.ndarray): Some signal
        b (jnp.ndarray): Some other signal with some content in common with a
        max_offset (int): Maximum expected offset of the signals
        correlation_size (int): Number of samples to use in each correlate
        a_center (int): Index in ``a`` around which to center the window. Default None, which means ``len(a)/2``
        b_center (int): Index in ``b`` around which to center the window. Default None, which means ``len(b)/2``
        chunk_size (int): Number of STFT frames to process in each chunk.

    Returns:
        - offset : int
            The sample offset of a relative to b
        - a_vs_b : float
            Correlation of ``a`` against ``b``
    """
    assert a.ndim == 2 and b.ndim == 2

    # Convert both to mono
    a = jnp.mean(a, axis=0)
    b = jnp.mean(b, axis=0)

    # Default values for the window centers
    if a_center is None:
        a_center = a.shape[0] // 2
    if b_center is None:
        b_center = b.shape[0] // 2
    # Avoid array out of bounds
    checkify.check(
        jnp.logical_and(
            a_center - max_offset - correlation_size // 2 >= 0,
            a_center + max_offset + correlation_size // 2 <= a.shape[0],
        ),
        "The window in a around a_center of size max_offset + correlation_size goes out of bounds",
    )
    checkify.check(
        jnp.logical_and(
            b_center - correlation_size // 2 >= 0,
            b_center + correlation_size // 2 <= b.shape[0],
        ),
        "The window in b around b_center of size max_offset + correlation_size goes out of bounds",
    )
    # Centered on a_center, extract 2*max_offset + correlation_size samples
    # (so offsets between -max_offset and max_offset)
    a_window = lax.dynamic_slice_in_dim(
        a,
        a_center - max_offset - correlation_size // 2,
        max_offset * 2 + 2 * (correlation_size // 2),
    )
    # From b, the sample window will only be correlation_size samples
    b_window = lax.dynamic_slice_in_dim(
        b, b_center - correlation_size // 2, 2 * (correlation_size // 2)
    )

    a_vs_b = fftconvolve_chunked(a_window, b_window[::-1], chunk_size=chunk_size)
    a_vs_b = a_vs_b[correlation_size // 2 : -correlation_size // 2]
    assert a_vs_b.shape[0] == a_window.shape[0] - 2 * (correlation_size // 2)
    # Compute offset of a relative to b
    offset = jnp.argmax(a_vs_b) - max_offset + (a_center - b_center)

    # Return offset and the max value of the correlation
    return offset, a_vs_b


def get_best_fs_ratio(
    a: jnp.ndarray,
    b: jnp.ndarray,
    max_drift: float,
    steps: int,
    max_offset: int,
    correlation_size: int,
    center: float = 1,
    verbose: bool = False,
) -> float:
    """
    Given two signals with components in common, tries to estimate the clock drift and offset of b vs a

    Args:
        a (jnp.ndarray): Some signal
        b (jnp.ndarray): Some other signal
        max_drift (float): Max sample rate drift, in percent, e.g. .02 = 2% clock drift
        steps (int): Number of sample rates to consider, between -max_drift and max_drift
        max_offset (int): Maximum expected offset of the signals
        correlation_size (int): Number of samples to use in each correlate
        center (float): Ratio to deviate from - default 1
        verbose: Verbosity, default False

    Returns:
        float: fs ratio to make ``b`` line up well with ``a``
    """
    b = np.array(b)
    assert a.ndim == 2 and b.ndim == 2
    # Sample rate ratios to try
    fs_ratios = center + jnp.linspace(-max_drift, max_drift, steps + 1)
    fs_ratios = fs_ratios.tolist()
    # The max correlation value for each fs ratio
    corr_max = []
    for n, ratio in enumerate(tqdm.tqdm(fs_ratios, desc="FS Ratios", disable=not verbose)):
        b_resampled = jnp.array(
            librosa.resample(b, orig_sr=1.0, target_sr=ratio), dtype=jnp.float64
        )
        # Compute the max correlation
        err, (_, corr) = align_over_window(a, b_resampled, max_offset, correlation_size)
        err.throw()
        corr_max.append(corr.max())

    corr_max = jnp.array(corr_max, dtype=jnp.float64)
    # Choose ratio with the highest correlation value
    return fs_ratios[jnp.argmax(corr_max)]


def apply_offsets_resample(
    b: jnp.ndarray, offset_locations: jnp.ndarray, offsets: jnp.ndarray, verbose: bool = False,
):
    """
    Adjust a signal b according to local offset estimations using resampling

    Args:
        b (jnp.ndarray): Some signal
        offset_locations (jnp.ndarray): Locations, in samples, of each local offset estimation
        offsets (jnp.ndarray): Local offset for the corresponding sample in offset_locations
        verbose: Verbosity, default False

    Returns:
        jnp.array: ``b`` with offsets applied
    """
    assert offset_locations.shape[0] == offsets.shape[-1]
    b = np.array(b)
    C, N = b.shape
    # Include signal boundaries in offset locations
    offset_locations = jnp.append(0, jnp.append(offset_locations, N - 100))
    # Allocate output signal
    output_size = int(
        jnp.sum(jnp.diff(offset_locations)) + int(jnp.max(jnp.abs(offsets)))
    )
    b_aligned = jnp.zeros((C, output_size), dtype=jnp.float64)
    # Set last offset to whatever the second to last one was
    offsets = jnp.concatenate([offsets, offsets[-1:]])

    current = 0
    for n, offset in enumerate(tqdm.tqdm(offsets, desc="Apply Offsets", disable=not verbose)):
        start = int(offset_locations[n])
        end = int(offset_locations[n + 1])
        # Compute the necessary resampling ratio to compensate for this offset
        ratio = 1 + (-offset + start - current) / (end - start)
        # Resample this portion of the signal, with some padding at the end
        # don't use audiotree.resample here because it doesn't work well when the greatest common denominator
        # of the two sample rates is large.
        resampled = librosa.resample(
            b[:, start : end + 100], orig_sr=1, target_sr=ratio
        )
        # Compute length and place the signal
        length = int(end - current - offset)
        b_aligned = b_aligned.at[:, current : current + length].set(
            resampled[..., :length]
        )
        current += length

    return b_aligned


def get_local_offsets(
    a: jnp.ndarray,
    b: jnp.ndarray,
    hop: int,
    max_offset: int,
    correlation_size: int,
    batch_size: int = 8,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Given two signals a and b, estimate local offsets to fix timing error of b relative to a

    Args:
        a (jnp.ndarray): Some signal
        b (jnp.ndarray): Some other signal appropriately zero-padded to be the same size as ``a``
        hop (int): Number of samples between successive offset estimations
        max_offset (int): Maximum expected offset of the signals
        correlation_size (int): Number of samples to use in each correlate
        batch_size (int): Number of samples to use in each batch

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Tuple composed of offset_locations and local_offsets.
            offset_locations - locations, in samples, of each local offset estimation.
            local_offsets - Estimates the best local offset for the corresponding sample in offset_locations.
    """
    # Compute the locations where we'll estimate offsets
    assert a.ndim == 2
    C, N = a.shape
    offset_locations = jnp.arange(
        correlation_size + max_offset, N - (correlation_size + max_offset), hop
    )

    local_offsets_list = []

    @jax.vmap
    def align_func(offset_location: jnp.ndarray) -> jnp.ndarray:
        err, (offset, _) = align_over_window(
            b,
            a,
            max_offset,
            correlation_size,
            offset_location,
            offset_location,
        )
        err.throw()
        return offset

    for i in tqdm.trange(
        math.ceil(len(offset_locations) / batch_size), desc="Offset Locations"
    ):
        batch_input = offset_locations[i * batch_size : (i + 1) * batch_size]
        # todo: pad batch_input to improve usage with vmap? Then undo the padding after calling.
        local_offsets_list.append(
            np.array(align_func(batch_input))
        )  # todo: cast to np.array or not?

    # Reshape results to match expected output shape
    local_offsets = jnp.concatenate(local_offsets_list, axis=0)

    assert local_offsets.ndim == 1
    assert local_offsets.shape == offset_locations.shape

    return offset_locations, local_offsets


@jax.vmap
def mult_by_best_filter_coefficients(M: jnp.ndarray, R: jnp.ndarray) -> jnp.ndarray:
    """
    Get the best vector H such that |M - HoR| is minimized, where M, H, R are complex. Then return H * R.

    Args:
        M (jnp.ndarray): STFT matrix, shape = (nbins, nframes)
        R (jnp.ndarray): STFT matrix, shape = (nbins, nframes)

    Returns:
        jnp.ndarray: H*R shaped ``(nbins, nframes)``
    """
    # Must be this datatype in order for minimize to work
    M = M.astype(jnp.complex128)
    R = R.astype(jnp.complex128)
    # 2 columns, one for real part, one for imag
    H = jnp.zeros((M.shape[0], 2), dtype=jnp.float64)

    @jax.vmap
    def get_new_filter_value_for_freq_bin(
        H: jnp.ndarray, M_i: jnp.ndarray, R_i: jnp.ndarray
    ) -> jnp.ndarray:
        def l1_sum(H_i: jnp.ndarray) -> jnp.ndarray:
            return jnp.abs(
                M_i.real
                + M_i.imag * 1j
                - (
                    H_i[0] * R_i.real
                    + H_i[0] * R_i.imag * 1j
                    + H_i[1] * R_i.real * 1j
                    - H_i[1] * R_i.imag
                )
            ).sum()

        # Note: The original numpy remixavier used "L-BFGS-B",
        #  and jax.scipy does have "l-bfgs-experimental-do-not-rely-on-this",
        #  but it doesn't work well for this case.
        method = "BFGS"

        out = minimize(
            l1_sum,
            H,
            # bounds=[(-100e100, 1e100), (-100e100, 1e100)],  # todo: use bounds if ever available in jax
            method=method,
        ).x
        return out

    H = get_new_filter_value_for_freq_bin(H, M, R)

    # Combine real and imaginary parts
    H_complex = (H[:, 0] + H[:, 1] * 1j).reshape(-1, 1)

    return H_complex * R


def median_filter(
    x: jnp.ndarray, window_size: int, axis: int = -1, mode="reflect"
) -> jnp.ndarray:
    # Compute half-window size (assumes window_size is odd)
    pad_width = window_size // 2

    # Build a pad specification for each dimension: pad only along the filtering axis.
    pad_spec = [(pad_width, pad_width) if i == axis else (0, 0) for i in range(x.ndim)]

    # Pad the input array to handle border effects.
    x_pad = jnp.pad(x, pad_spec, mode=mode)

    # Get the original length along the filtering axis.
    orig_len = x.shape[axis]

    # Build a list of sliding windows along the axis.
    # Each slice takes a segment of length orig_len starting at a shifted index.
    windows = [
        jnp.take(x_pad, jnp.arange(i, i + orig_len), axis=axis)
        for i in range(window_size)
    ]

    # Stack the windows along a new axis (here, the last axis).
    windows = jnp.stack(windows, axis=-1)

    # Compute and return the median along the new window axis.
    return jnp.median(windows, axis=-1)


def remove_outliers(x: jnp.ndarray, median_size: int = 13) -> jnp.ndarray:
    """
    Replaces any points in the vector x which lie outside of one std dev of the local median

    Args:
        x (jnp.ndarray): Input vector to clean
        median_size (int): Median filter size, default 13

    Returns:
        jnp.ndarray: Cleaned version of ``x``
    """
    assert x.ndim == 1
    N = x.shape[0]

    # Convert to NumPy for median filter
    median_filtered = median_filter(x, median_size)
    global_std = jnp.std(x)

    def func(prev_v: jnp.ndarray, xs):
        cur_v, filtered_slice = xs
        new_val = jnp.where(jnp.abs(cur_v - filtered_slice) > global_std, prev_v, cur_v)

        return new_val, new_val

    _, x_smoothed = lax.scan(
        func, x[0], (x[1:], median_filtered[1:]), length=N - 1, unroll=False
    )

    x = x.at[1:].set(x_smoothed)

    return x


@argbind.bind()
def wiener_enhance(
    target: jnp.ndarray,
    accomp: jnp.ndarray,
    thresh: float = -6,
    transit: float = 3,
    n_fft: int = 2048,
    verbose: bool = False,
):
    """
    Given a noisy signal and a signal which approximates the noise, try to remove the noise.

    Args:
        target (jnp.ndarray): Noisy signal
        accomp (jnp.ndarray): Approximate noise
        thresh (float): Sigmoid threshold, default -6
        transit (float): Sigmoid transition, default 3
        n_fft (int): FFT length, default 2048 (hop is always n_fft/
        verbose: Verbose mode, default False

    Returns:
        jnp.ndarray: ``target`` after wiener filter that tried to remove noise.
    """
    target_spec, accomp_spec = stft_in_batches(
        jnp.stack([target, accomp], axis=0), n_fft=n_fft, hop_length=n_fft // 4, verbose=verbose,
    )

    spec_ratio = amplitude_to_db(jnp.abs(target_spec)) - amplitude_to_db(jnp.abs(accomp_spec))  # fmt: skip
    spec_ratio = (spec_ratio - thresh) / transit
    mask = 0.5 + 0.5 * (spec_ratio / jnp.sqrt(1 + spec_ratio**2))

    filtered = istft(target_spec * mask, hop_length=n_fft // 4)

    return filtered


def pad(a: jnp.ndarray, b: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Given two vectors, pad the shorter one with zeros (at the end) so that they are the same size

    Args:
        a (jnp.ndarray): vector shaped ``(C, Na)``
        b (jnp.ndarray): vector shaped ``(C, Nb)``

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: ``a_padded`` and ``b_padded``
    """
    Ca, Na = a.shape
    Cb, Nb = b.shape
    assert Ca == Cb

    if Na > Nb:
        b = jnp.zeros_like(a).at[:, :Nb].set(b)
    elif Na < Nb:
        a = jnp.zeros_like(b).at[:, :Na].set(a)
    return a, b


@argbind.bind()
def align(
    a: jnp.ndarray,
    b: jnp.ndarray,
    fs: int,
    correlation_size: float = 4.0,
    max_global_offset: float = 2.0,
    max_skew_offset: float = 2.0,
    max_skew: float = 0.02,
    hop: float = 0.2,
    max_local_offset: float = 0.1,
    batch_size: int = 128,
    skip_local_offsets: bool = False,
    verbose: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Aligns signal ``b`` to ``a`` by fixing global offset, finding optimal resampling rate and fixing local offsets.

    Args:
        a (jnp.ndarray): Some signal
        b (jnp.ndarray): Some other signal to align to a
        fs (int): Sampling rate of the signals a and b
        correlation_size (float): Size, in seconds, of windows over which correlations will be taken, default 4
        max_skew_offset (float): Size, in seconds, of the maximum offset to consider for skew estimation
        max_global_offset (float): Length, in seconds, of the largest global offset to consider, default 2
        max_skew (float): Maximum percentage skew to consider, default .02
        hop (float): Time, in seconds, between successive offset estimations, default .2
        max_local_offset (float): Maximum offset in seconds for local each offset estimation, default .1
        batch_size (int): Batch size
        verbose: Verbosity with tqdm

    Returns:
        - a_aligned : np.ndarray
            The signal ``a``, unchanged except for possibly zero-padding
        - b_aligned : np.ndarray
            The signal ``b``, aligned in time to match a
    """
    # Make them the same length
    a, b = pad(a, b)
    C, N = a.shape
    # Fix any global offset
    if max_global_offset > 0:
        err, (offset, _) = align_over_window(
            a,
            b,
            max_offset=int(max_global_offset * fs),
            correlation_size=int(correlation_size * fs),
        )
        err.throw()
        print(f"Global offset detected: {offset} samples ({offset/fs:.4f} seconds)")
        if offset < 0:
            a = jnp.append(jnp.zeros((C, -offset), dtype=a.dtype), a, axis=1)
        elif offset > 0:
            b = jnp.append(jnp.zeros((C, offset), dtype=a.dtype), b, axis=1)
    # Fix skew
    if max_skew > 0:
        # Downsample to 2 kHz for speed! Doesn't make a big difference performance-wise
        if fs > 2000:
            fs_ds = 2000
            a_ds = resample(a, old_sr=fs, new_sr=fs_ds)
            b_ds = resample(b, old_sr=fs, new_sr=fs_ds)
        else:
            a_ds = a.copy()
            b_ds = b.copy()
            fs_ds = fs
        if max_skew > 0.0001:
            # Get coarse estimate of best sampling rate
            fs_ratio = get_best_fs_ratio(
                a_ds,
                b_ds,
                max_skew,
                200,
                int(max_skew_offset * fs_ds),
                int(correlation_size * fs_ds),
                verbose=verbose,
            )
        else:
            fs_ratio = 1.0
        # Get fine estimate
        fs_ratio = get_best_fs_ratio(
            a_ds,
            b_ds,
            0.0001,
            200,
            int(max_skew_offset * fs_ds),
            int(correlation_size * fs_ds),
            fs_ratio,
            verbose=verbose,
        )
        print(f"Sample rate ratio detected: {fs_ratio:.6f} ({(fs_ratio-1)*100:.4f}% skew)")

        skew_threshold = 0.0001
        if abs(fs_ratio - 1.0) < skew_threshold:
            print(f"Skew below threshold ({skew_threshold*100:.2f}%), skipping resample to avoid artifacts")
        else:
            print(f"Applying sample rate correction...")
            b = jnp.array(
                librosa.resample(np.array(b), orig_sr=1, target_sr=fs_ratio),
                dtype=jnp.float64,
            )

    # Estimate offset locations every "hop" seconds
    if skip_local_offsets:
        print("Skipping local offset correction")
    else:
        offset_locations, offsets = get_local_offsets(
            a,
            b,
            int(fs * hop),
            int(fs * max_local_offset),
            int(fs * correlation_size),
            batch_size=batch_size,
        )

        # Remove any big jumps in the offset list
        offsets = remove_outliers(offsets)
        # Adjust source according to these offsets
        b = apply_offsets_resample(b, offset_locations, offsets, verbose=verbose)

    # Make sure they are the same length
    a, b = pad(a, b)

    return a, b


def reverse_channel(
    a: jnp.ndarray,
    b: jnp.ndarray,
    n_fft: int = 2**13,
    win_length: int = 2**12,
    hop_length: int = 2**10,
):
    """
    Estimates the channel distortion in b relative to a and reverses it

    WARNING: This JAX implementation has known issues:
    - Uses unbounded BFGS optimization instead of bounded L-BFGS-B (original)
    - JAX's 'l-bfgs-experimental' doesn't work well for this case
    - Can produce extreme filter coefficients causing artifacts or silence
    - NOT recommended for official stems (use --skip_reverse_channel)
    - Only use for user-recorded stems with actual channel distortion

    The original numpy implementation uses scipy's L-BFGS-B with bounds
    [(-1e100, 1e100), (-1e100, 1e100)] which keeps filter coefficients stable.
    JAX's minimize() doesn't support bounds parameter.

    Args:
        a (jnp.ndarray): Some signal
        b (jnp.ndarray): Some other signal with channel distortion relative to a
        n_fft (int): Number of samples in each FFT computation, default 2**13
        win_length (int): Number of samples in each window, default 2**12
        hop_length (int): Number of samples between successive FFT computations, default 2**10

    Returns:
        jnp.ndarray: The signal ``b``, filtered to reduce channel distortion
    """
    assert a.ndim == 2
    C = a.shape[0]

    # Compute spectrograms
    a_spec, b_spec = stft_in_batches(
        jnp.stack([a, b], axis=0),
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )
    assert a_spec.shape[0] == C
    # Process spectrograms
    b_spec_filtered = mult_by_best_filter_coefficients(a_spec, b_spec)

    # Get back to time domain
    b_filtered = istft(b_spec_filtered, win_length=win_length, hop_length=hop_length)

    return b_filtered
