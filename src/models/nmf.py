import numpy as np
import librosa
import scipy
import sklearn
from dlex import MainConfig
from dlex.datasets.torch import Dataset
from tqdm import tqdm

from ..datasets.mir_eval.separation import bss_eval_sources
from sklearn.base import BaseEstimator


def bss_eval_global(mixed_wav, src1_wav, src2_wav, pred_src1_wav, pred_src2_wav):
    len_cropped = pred_src1_wav.shape[-1]
    src1_wav = src1_wav[:, :len_cropped]
    src2_wav = src2_wav[:, :len_cropped]
    mixed_wav = mixed_wav[:, :len_cropped]
    gnsdr = np.zeros(2)
    gsir = np.zeros(2)
    gsar = np.zeros(2)
    total_len = 0
    for i in range(len(mixed_wav)):
        sdr, sir, sar, _ = bss_eval_sources(
            np.array([src1_wav[i], src2_wav[i]]),
            np.array([pred_src1_wav[i], pred_src2_wav[i]]), False)
        sdr_mixed, _, _, _ = bss_eval_sources(
            np.array([src1_wav[i], src2_wav[i]]),
            np.array([mixed_wav[i], mixed_wav[i]]), False)

        nsdr = sdr - sdr_mixed
        gnsdr += len_cropped * nsdr
        gsir += len_cropped * sir
        gsar += len_cropped * sar
        total_len += len_cropped

    gnsdr = gnsdr / total_len
    gsir = gsir / total_len
    gsar = gsar / total_len
    return gnsdr, gsir, gsar


class NMF(BaseEstimator):
    def __init__(self, params: MainConfig, dataset: Dataset):
        super().__init__()
        self.params = params
        self.configs = params.dataset
        self.dataset = dataset
        self.nmf = sklearn.decomposition.NMF(
            self.configs.num_components, tol=1e-6, solver="mu", beta_loss=self.configs.beta_loss)
        self.kmeans = sklearn.cluster.KMeans(
            2)

    def fit(self, X, y):
        pass

    def fit_transform(self, X, y=None):
        print(X)

    def score(self, X, Y, metric):
        components = [[] for _ in range(2)]
        for i in tqdm(range(len(X)), desc="Processing"):
            x, y = X[i], Y[i]
            x += np.random.rand(*x.shape) * 0.001
            S = librosa.stft(x)
            A = np.absolute(S)
            W = self.nmf.fit_transform(A)
            self.kmeans.fit(W.T)
            labels = self.kmeans.labels_.tolist()
            print(labels)
            H = self.nmf.components_

            comp = [scipy.zeros(len(x)), scipy.zeros(len(x))]
            for n in range(self.configs.num_components):
                y = scipy.outer(W[:, n], H[n]) * np.exp(1j * np.angle(S))
                y = librosa.istft(y)
                comp[labels[n]] += y
            components[0].append(comp[0])
            components[1].append(comp[1])
            # components[0].append(x / 2)
            # components[1].append(x / 2)

        gnsdr1, gsir1, gsar1 = bss_eval_global(
            X, Y[:, 0, :], Y[:, 1, :], np.array(components[0]), np.array(components[1]))
        gnsdr2, gsir2, gsar2 = bss_eval_global(
            X, Y[:, 0, :], Y[:, 1, :], np.array(components[0]), np.array(components[1]))

        gnsdr = max(np.average(gnsdr1), np.average(gnsdr2))
        gsir = max(np.average(gsir1), np.average(gsir2))
        gsar = max(np.average(gsar1), np.average(gsar2))

        return gnsdr, gsir, gsar

        if metric == "gnsdr":
            return gnsdr
        elif metric == "gsir":
            return gsir
        elif metric == "gsar":
            return gsar