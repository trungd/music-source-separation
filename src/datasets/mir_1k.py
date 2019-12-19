import random
import os

import numpy as np
import librosa
from dlex import MainConfig
from dlex.datasets import DatasetBuilder
from dlex.datasets.sklearn import SklearnDataset


def _sample_range(wav, sr, duration):
    assert(wav.ndim <= 2)

    target_len = int(sr * duration)
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def _pad_wav(wav, sr, duration):
    assert(wav.ndim <= 2)

    n_samples = int(sr * duration)
    pad_len = np.maximum(0, n_samples - wav.shape[-1])
    if wav.ndim == 1:
        pad_width = (0, pad_len)
    else:
        pad_width = ((0, 0), (0, pad_len))
    wav = np.pad(wav, pad_width=pad_width, mode='constant', constant_values=0)

    return wav


def get_random_wav(filenames, sec, sr=16000):
    # load wav -> pad if necessary to fit sr*sec -> get random samples with len = sr*sec -> map = do this for all in filenames -> put in np.array
    src1_src2 = np.array(list(
        map(lambda f: _sample_range(_pad_wav(librosa.load(f, sr=sr, mono=False)[0], sr, sec), sr, sec), filenames)))
    mixed = np.array(list(map(lambda f: librosa.to_mono(f), src1_src2)))
    src1, src2 = src1_src2[:, 0], src1_src2[:, 1]
    return mixed, src1, src2


class MIR_1K(DatasetBuilder):
    def __init__(self, params: MainConfig):
        super().__init__(
            params,
            downloads=["http://mirlab.org/dataset/public/MIR-1K.rar"])

    def maybe_preprocess(self, force=False):
        pass

    def get_sklearn_wrapper(self, mode: str):
        return SklearnMIR_1K(self)


class SklearnMIR_1K(SklearnDataset):
    def __init__(self, builder):
        super().__init__(builder)

        wavfiles = []
        for (root, dirs, files) in os.walk(os.path.join(builder.get_raw_data_dir(), "MIR-1K", "Wavfile")):
            wavfiles.extend(['{}/{}'.format(root, f) for f in files if f.endswith(".wav")])
        wavfiles = random.sample(wavfiles, self.configs.size)
        mixed, src1, src2 = get_random_wav(wavfiles, self.configs.seconds, 16000)

        self.init_dataset(mixed, np.stack([src1, src2], axis=1))