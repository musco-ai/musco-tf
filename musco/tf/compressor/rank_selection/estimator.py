import numpy as np
import tensorly as tl
from musco.tf.compressor.rank_selection import vbmf

tl.set_backend("numpy")


def weaken_rank(rank, extreme_rank, k):
    if rank < 21:
        wrank = rank
    elif extreme_rank == 0:
        wrank = rank
        # print("EVBMF returned 0 rank.")
    else:
        wrank = int(rank - k * (rank - extreme_rank))

    return wrank


def estimate_vbmf_ranks(weights, k=1):
    """ Unfold the 2 modes of the Tensor the decompositions will
    be performed on, and estimates the ranks of the matrices using VBMF.
    """

    if len(weights.shape) > 2:
        unfold_0 = tl.base.unfold(weights, 0)
        unfold_1 = tl.base.unfold(weights, 1)

        try:
            _, diag_0, _, _ = vbmf.evbmf(unfold_0)
        except:
            _, diag_0, _, _ = vbmf.vbmf(unfold_0)

        try:
            _, diag_1, _, _ = vbmf.evbmf(unfold_1)
        except:
            _, diag_1, _, _ = vbmf.evbmf(unfold_1)

        ranks = [diag_0.shape[0], diag_1.shape[1]]

        ranks_weak = [weaken_rank(unfold_0.shape[0], ranks[0], k),
                      weaken_rank(unfold_1.shape[0], ranks[1], k)]

    else:
        unfold = weights.data
        unfold = unfold.numpy()

        try:
            _, diag, _, _ = vbmf.evbmf(unfold)
        except:
            try:
                _, diag, _, _ = vbmf.evbmf(unfold.T)
            except:
                try:
                    _, diag, _, _ = vbmf.vbmf(unfold)
                except:
                    _, diag, _, _ = vbmf.vbmf(unfold.T)

        rank = diag.shape[0]
        ranks_weak = weaken_rank(min(unfold.shape), rank, k)

    return ranks_weak


def count_cp4_parameters(tensor_shape, rank=8):
    cout, cin, kh, kw = tensor_shape
    cp4_count = rank * (cin + kh + kw + cout)

    return cp4_count


def count_cp3_parameters(tensor_shape, rank=8):
    cout, cin, kh, kw = tensor_shape
    cp3_count = rank * (cin + kh * kw + cout)

    return cp3_count


def count_tucker2_parameters(tensor_shape, ranks=(8, 8)):
    cout, cin, kh, kw = tensor_shape

    if type(ranks) != list or type(ranks) != tuple:
        ranks = [ranks, ranks]

    tucker2_count = ranks[-2] * cin + np.prod(ranks[-2:]) * kh * kw + ranks[-1] * cout

    return np.array(tucker2_count)


def count_parameters(tensor_shape, rank=None, key="cp3"):
    params_count = None

    if key == "cp4":
        params_count = count_cp4_parameters(tensor_shape, rank=rank)
    elif key == "cp3":
        params_count = count_cp3_parameters(tensor_shape, rank=rank)
    elif key == "tucker2":
        params_count = count_tucker2_parameters(tensor_shape, ranks=rank)

    return params_count


def estimate_rank_for_compression_rate(tensor_shape, rate=2, key="tucker2"):
    """
        Find max rank for which inequality (initial_count / decomposition_count > rate) holds true
    """

    initial_count = np.prod(tensor_shape)

    if key != "svd":
        cout, cin, kh, kw = tensor_shape

    if key == "cp4":
        max_rank = initial_count // (rate * (cin + kh + kw + cout))
    elif key == "cp3":
        max_rank = initial_count // (rate * (cin + kh * kw + cout))
    elif key == "tucker2":
        # tucker2_rank when R4=beta*R3.
        if cout > cin:
            beta = 1.6
        else:
            beta = 1.

        a = 1
        b = (cin + beta * cout) / (beta * kh * kw)
        c = -cin * cout / rate / beta
        discr = b ** 2 - 4 * a * c
        max_rank = int((-b + np.sqrt(discr)) / 2 / a)
        # [R4, R3].
        max_rank = (int(beta * max_rank), max_rank)
    elif key == 'svd':
        max_rank = initial_count // (rate * sum(tensor_shape[:2]))

    return max_rank
