# Remixavier-JAX

This is a JAX port of some of [Remixavier](https://github.com/craffel/remixavier).

> C. Raffel and D. P. W. Ellis, ["Estimating Timing and Channel Distortion Across Related Signals"](http://colinraffel.com/publications/icassp2014estimating.pdf), Proceedings of the 2014 IEEE International Conference on Acoustics, Speech and Signal Processing, 2014.

If you have a full mixture ("vocals"+"instrumental") of a song and a component of the song ("vocals"), this project can help you get the remainder ("instrumental"). It will produce good results even if the input mixture and the component are misaligned in time or in volume.

## Install requirements

```bash
pip3 install jax[cuda]
pip3 install -r requirements.txt
```

## Demo:

Basic usage:
```bash
python3 demo.py --mix_file="mixture.wav" --source_file="vocals.wav" --output_path="instrumental.wav" --duration=10.0
```

Help:
```bash
python3 demo.py --help
```

## Notes

### Working with Official Stems

For official artist-released stems, the default settings work best:

```bash
python3 demo.py \
  --mix_file="full_mix.flac" \
  --source_file="instrumental.flac" \
  --output_path="vocals.wav"
```

**Note**: `reverse_channel` correction is **disabled by default** due to a known issue in the JAX port. The original uses bounded L-BFGS-B optimization, but JAX's `minimize()` doesn't support bounds, causing unbounded BFGS to produce extreme filter coefficients and artifacts.

Only enable reverse_channel (`--apply_reverse_channel`) if you have user-recorded stems with actual channel distortion (different EQ/compression between stems).

The algorithm automatically:
- Skips sample rate correction if skew < 0.01% (reduces artifacts)
- Detects and corrects global timing offset
- Applies local offset corrections for timing drift
- Skips reverse_channel by default (prevents JAX BFGS artifacts)

### GPU Memory Issues

If you encounter CUDA out-of-memory errors, you can reduce the chunk size (default: `2**12` = 4096) to lower GPU memory usage:

```bash
python3 demo.py --mix_file="mixture.wav" --source_file="vocals.wav" --output_path="instrumental.wav" --align_over_window.chunk_size=2048
```

Smaller chunk sizes use less GPU memory but process more slowly. Always use a power of 2.
