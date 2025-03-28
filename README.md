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

```bash
python3 demo.py --mix_file="mixture.wav" --source_file="vocals.wav" --duration=10.0
```

Help:
```bash
python3 demo.py --help
```
