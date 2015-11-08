'''
Created on Nov 1, 2015

@author: alexey
'''


from sklearn.cluster import MiniBatchKMeans
from nd.utils import time_it

import logging
log = logging.getLogger('nd.algorithms')


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

