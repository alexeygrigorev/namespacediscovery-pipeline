'''
Created on Nov 8, 2015

@author: alexey
'''

from sklearn.decomposition import randomized_svd
from sklearn.decomposition import NMF
from sklearn.random_projection import GaussianRandomProjection


from sklearn.preprocessing import Normalizer
normalizer = Normalizer(copy=False)

from nd.utils import time_it

import logging
log = logging.getLogger('nd.algorithms')

def svd(X, K):
    _, _, Vt = randomized_svd(X, n_components=K)
    X_red = X.dot(Vt.T)
    X_red = normalizer.fit_transform(X_red)
    return X_red


def nmf(X, K):
    nmf = NMF(n_components=K)
    X_red = nmf.fit_transform(X)
    X_red = normalizer.fit_transform(X_red)
    return X_red

def random(X, K):
    grp = GaussianRandomProjection(n_components=K)
    X_red = grp.fit_transform(X)
    X_red = normalizer.fit_transform(X_red)
    return X_red

@time_it
def dimred(vectorized, algo, N, **kwargs):
    X = vectorized.X

    if algo == 'svd':
        X_red = svd(X, N)
    elif algo == 'nmf':
        X_red = nmf(X, N)
    elif algo == 'random':
        X_red = random(X, N)
    else:
        raise Exception('unknown algorithm')

    vectorized.X = X_red
    return vectorized

