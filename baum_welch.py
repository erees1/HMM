import numpy as np
from HMM.fb import HMMfb, pairwise


def baum_welch(v, Tr, E, p1, eps=1e-2):
    """Run Baum Welch (EM) algorithm on HMM, learns hidden states and
    the transition matrix Tr

    Args:
        v (np.array): Observations, shape (T,)
        Tr (np.array): Transistion matrix: Tr_ij = p(ht+1 = i | ht = j)
        E (np.array): Emission matrix: E_ij = p(vt = i | ht = j)
        p1 (np.array): Initial state, shape (n,)
        eps (float, optional): Desired error in logspace. Defaults to 1e-2.

    Returns:
        logliks, Tr, gamma
    """

    T = len(v)
    pair_probs = np.zeros((T,) + Tr.shape)
    logliks = []
    step = 0

    while True:
        # E step
        gamma, alpha, beta, loglik = HMMfb(v, Tr, E, p1)

        for t in range(T - 1):
            # pairwise returns ht+1 in rows, ht in columns
            pair_probs[t, :, :] = pairwise(t, v, alpha, beta, E, Tr)

        # M Step
        new_Tr = pair_probs.sum(axis=0) / pair_probs.sum(axis=0).sum(axis=1)
        Tr = new_Tr

        logliks.append((loglik))
        if len(logliks) > 2 and abs(loglik - logliks[-2]) < eps:
            break
        else:
            step += 1
            print(loglik)

    return logliks, Tr, gamma
