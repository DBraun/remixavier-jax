import os
from pathlib import Path
from typing import Tuple, Union

import argbind
import librosa
import numpy as np
from scipy.io import wavfile

from estimate import pad, align, reverse_channel, wiener_enhance, jnp

align = argbind.bind(align)
wiener_enhance = argbind.bind(wiener_enhance)

PathLike = Union[Path, str, int, os.PathLike]


def load_audio(
    filepath: PathLike, sr: float = None, duration: float = None
) -> Tuple[jnp.ndarray, int]:
    x, fs = librosa.load(filepath, mono=False, sr=sr, duration=duration)
    x = jnp.array(x, dtype=jnp.float64)
    if x.ndim == 1:
        x = jnp.expand_dims(x, axis=0)
    return x, fs


@argbind.bind(without_prefix=True)
def main(
    mix_file: PathLike = "mixture.wav",
    source_file: PathLike = "vocals.wav",
    output_path: PathLike = "output.wav",
    duration: float = None,
):
    print(f"Loading: {mix_file}")
    mix, fs = load_audio(mix_file, sr=None, duration=duration)
    print(f"Loading: {source_file}")
    source, fs = load_audio(source_file, sr=fs, duration=duration)

    print("Aligning...")
    mix, source = align(mix, source, fs=fs)
    print("Subtracting...")
    source = reverse_channel(mix, source)
    mix, source = pad(mix, source)
    print("Enhancing...")
    subtracted = wiener_enhance(mix - source, source)

    subtracted = np.array(subtracted)
    wavfile.write(output_path, fs, subtracted.T)


if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()
