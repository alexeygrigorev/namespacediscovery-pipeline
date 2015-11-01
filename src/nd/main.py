# -*- coding: utf-8 -*-


'''
Created on Nov 1, 2015

@author: alexey
'''

import logging
logging.basicConfig()
logging.getLogger('nd.etl.read').setLevel(logging.DEBUG)
logging.getLogger('nd.etl.ivs').setLevel(logging.DEBUG)
logging.getLogger('nd.cluster').setLevel(logging.DEBUG)

import nd.etl.read as read
import nd.etl.ivs as ivs
import nd.cluster.algos as clustering


props = {'input': '/media/alexey/B604C15104C11571/tmp/mlp/mlp-output/', 
         'categories': '/media/alexey/B604C15104C11571/tmp/mlp/category_info_refined.txt'}
result = read.main_read(props)


result = ivs.process(ivs.IDV_NO_DEF, result)
# TODO: make these params configurable
vectorized = ivs.vectorize(result, use_idf=True, sublinear_tf=True)

labels = clustering.cluster(vectorized, algo='kmeans', k=10000)
