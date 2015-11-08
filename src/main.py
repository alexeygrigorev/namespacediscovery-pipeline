# -*- coding: utf-8 -*-


'''
Created on Nov 1, 2015

@author: alexey
'''

import logging
logging.basicConfig()
logging.getLogger('nd.read').setLevel(logging.DEBUG)
logging.getLogger('nd.algorithms').setLevel(logging.DEBUG)
logging.getLogger('nd.utils').setLevel(logging.DEBUG)


from nd.read import mlp_read, scheme

import nd.algorithms.ivs as ivs

from nd.algorithms.cluster import cluster
from nd.algorithms.dimred import dimred
from nd.algorithms.evaluate import create_evaluator



props = {'input': '/media/alexey/B604C15104C11571/tmp/mlp/mlp-output/', 
         'categories': '/media/alexey/B604C15104C11571/tmp/mlp/category_info_refined.txt'}
mlp_data = mlp_read.main_read(props)
evaluator = create_evaluator(mlp_data)

mlp_data = ivs.process(ivs.IDV_NO_DEF, mlp_data)


vectors = ivs.vectorize(mlp_data, use_idf=True, sublinear_tf=True)
vectors = dimred(vectors, 'svd', N=100)

labels = cluster(vectors, algo='kmeans', k=1000)

need_print = False

min_size = 3
purity_threshold = 0.8
identifier = u'μ'

pure_clusters = evaluator.high_purity_clusters(labels, threshold=purity_threshold,
                              min_size=min_size, all_categories=1)

if need_print:

    evaluator.describe(labels)

    evaluator.report_overall(labels, purity_threshold=purity_threshold,
                             min_size=min_size, sort_by='size')

    evaluator.find_identifier(labels, purity_threshold=purity_threshold, id=identifier, 
                              min_size=min_size, collection_weighting=0)


    for cl in pure_clusters:
        cluster_id = cl['cluster']
        evaluator.print_cluster(labels, cluster_id, sort_by_score=1, normalize_score=1)


res_root = '/home/alexey/git/namespacediscovery-pipeline/data/classification/';

msc = scheme.read('msc', res_root + 'msc.txt')
pacs = scheme.read('pacs', res_root + 'pacs.txt')
acm = scheme.read('acm', res_root + 'acm_skos_taxonomy.xml')

merged_scheme = scheme.merge([msc, pacs, acm])


from nd.algorithms.namespace import assign_clusters_to_scheme

ROOT = assign_clusters_to_scheme(merged_scheme, labels, mlp_data, evaluator,
                                 pure_clusters)

ROOT.print_ns(evaluator, print_rels=1)
