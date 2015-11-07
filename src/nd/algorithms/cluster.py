'''
Created on Nov 1, 2015

@author: alexey
'''

from sklearn.decomposition import randomized_svd
from sklearn.cluster import MiniBatchKMeans

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


@time_it
def dimred(vectorized, algo, **kwargs):
    X = vectorized.X

    if algo == 'svd':
        assert 'N' in kwargs
        X_red = svd(X, kwargs['N'])
        vectorized.X = X_red
    else:
        raise Exception('unknown algorithm')

    return vectorized


def k_means(X, k):
    log.debug('applying KMeans with k=%d...' % k)
    km = MiniBatchKMeans(n_clusters=k, init_size=k*3, n_init=10, init='random')
    km.fit(X)
    log.debug('done')
    return km.labels_


@time_it
def cluster(vectorized, algo, **kwargs):
    if algo == 'kmeans':
        return k_means(vectorized.X, **kwargs)
    return None

