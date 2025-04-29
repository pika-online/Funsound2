from funsound.utils import *
from sklearn.cluster._kmeans import k_means

class SpectralCluster:
    r"""A spectral clustering mehtod using unnormalized Laplacian of affinity matrix.
    This implementation is adapted from https://github.com/speechbrain/speechbrain.
    """

    def __init__(self, min_num_spks=1, max_num_spks=15, pval=0.022):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.pval = pval

    def __call__(self, X, oracle_num=None):
        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)

        # Refining similarity matrix with pval
        prunned_sim_mat = self.p_pruning(sim_mat)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)

        # Perform clustering
        labels = self.cluster_embs(emb, num_of_spk)

        return labels

    def get_sim_mat(self, X):
        # Cosine similarities
        M = sklearn.metrics.pairwise.cosine_similarity(X, X)
        return M

    def p_pruning(self, A):
        if A.shape[0] * self.pval < 6:
            pval = 6.0 / A.shape[0]
        else:
            pval = self.pval

        n_elems = int((1 - pval) * A.shape[0])

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0
        return A

    def get_laplacian(self, M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle=None):
        lambdas, eig_vecs = scipy.linalg.eigh(L)

        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(
                lambdas[self.min_num_spks - 1 : self.max_num_spks + 1]
            )
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        _, labels, _ = k_means(emb, k)
        return labels

    def getEigenGaps(self, eig_vals):
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list


def merge_by_cos( labels, embs, cos_thr):
        # merge the similar speakers by cosine similarity
        assert cos_thr > 0 and cos_thr <= 1
        while True:
            spk_num = labels.max() + 1
            if spk_num == 1:
                break
            spk_center = []
            for i in range(spk_num):
                spk_emb = embs[labels == i].mean(0)
                spk_center.append(spk_emb)
            assert len(spk_center) > 0
            spk_center = np.stack(spk_center, axis=0)
            norm_spk_center = spk_center / np.linalg.norm(spk_center, axis=1, keepdims=True)
            affinity = np.matmul(norm_spk_center, norm_spk_center.T)
            affinity = np.triu(affinity, 1)
            spks = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[spks] < cos_thr:
                break
            for i in range(len(labels)):
                if labels[i] == spks[1]:
                    labels[i] = spks[0]
                elif labels[i] > spks[1]:
                    labels[i] -= 1
        return labels

class FBank(object):
    def __init__(self, n_mels, sample_rate, mean_nor: bool = False):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

        self.opts = knf.FbankOptions()
        self.opts.frame_opts.samp_freq = sample_rate
        self.opts.mel_opts.num_bins = n_mels
        self.opts.frame_opts.dither = 0.0  # 默认不加扰动

    def __call__(self, wav: np.ndarray, dither=0.0) -> np.ndarray:
        assert self.sample_rate == 16000, "只支持16kHz音频"
        assert isinstance(wav, np.ndarray), "输入应为 NumPy 数组"

        # 处理多通道音频：取第一个通道
        if wav.ndim == 2:
            wav = wav[0, :]
        assert wav.ndim == 1, "音频必须是一维或[1, T]"

        # 更新扰动参数
        self.opts.frame_opts.dither = dither

        # 实例化 fbank 提取器
        fbank = knf.OnlineFbank(self.opts)
        fbank.accept_waveform(self.sample_rate, wav.astype(np.float32))
        fbank.input_finished()

        # 提取所有帧
        feats = [fbank.get_frame(i) for i in range(fbank.num_frames_ready)]
        feats = np.stack(feats)  # 形状: [T, N]

        if self.mean_nor:
            feats = feats - np.mean(feats, axis=0, keepdims=True)

        return feats  # 返回 NumPy 数组 [T, n_mels]