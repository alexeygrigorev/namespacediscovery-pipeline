# -*- coding: utf-8 -*-


'''
Created on Nov 1, 2015

@author: alexey
'''

import logging
logging.basicConfig()
logging.getLogger('nd.etl.read').setLevel(logging.DEBUG)
logging.getLogger('nd.etl.ivs').setLevel(logging.DEBUG)
logging.getLogger('nd.algorithms').setLevel(logging.DEBUG)
logging.getLogger('nd.utils').setLevel(logging.DEBUG)


import nd.etl.read as read
import nd.etl.ivs as ivs

from nd.algorithms.cluster import cluster, dimred
from nd.algorithms.evaluate import create_evaluator


props = {'input': '/media/alexey/B604C15104C11571/tmp/mlp/mlp-output/', 
         'categories': '/media/alexey/B604C15104C11571/tmp/mlp/category_info_refined.txt'}
data = read.main_read(props)
evaluator = create_evaluator(data)


data = ivs.process(ivs.IDV_NO_DEF, data)


vectors = ivs.vectorize(data, use_idf=True, sublinear_tf=True)
vectors = dimred(vectors, 'svd', N=100)

labels = cluster(vectors, algo='kmeans', k=1000)

evaluator.describe(labels)


evaluator.report_overall(labels, purity_threshold=0.8, min_size=3, sort_by='size')


