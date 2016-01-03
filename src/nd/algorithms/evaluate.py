# -*- coding: utf-8 -*-

'''
Created on Oct 18, 2015

@author: alexey
'''



import numpy as np
import operator 
from collections import Counter, defaultdict
from itertools import groupby

from fuzzywuzzy import fuzz, process


def create_evaluator(data):
    return Evaluator(data.document_titles, data.document_identifiers, 
        data.document_relations, data.document_categories)

class Evaluator():
    doc_titles = None
    doc_ids = None
    doc_ids_definitions = None
    doc_categories = None

    def __init__(self, doc_titles, doc_ids, doc_ids_definitions, doc_categories):
        self.doc_titles = doc_titles
        self.doc_ids = doc_ids
        self.doc_ids_definitions = doc_ids_definitions
        self.doc_categories = doc_categories

    def cluster_purity(self, indices):
        if indices.dtype == 'bool':
            indices, = np.where(indices)

        size = len(indices)
        if size == 0:
            return 0.0, ''

        cluster_cats = Counter()

        for idx in indices:
            cluster_cats.update(self.doc_categories[idx])

        if len(cluster_cats) == 0:
            return 0.0, ''

        category, cat_cnt = cluster_cats.most_common()[0]
        return 1.0 * cat_cnt / size, category

    def cluster_categories(self, indices):
        if indices.dtype == 'bool':
            indices, = np.where(indices)

        cluster_cats = Counter()
        for idx in indices:
            cluster_cats.update(self.doc_categories[idx])

        return cluster_cats

    def high_purity_clusters(self, cluster_assignment, threshold, min_size=5, 
                             k=None, all_categories=0):
        if not k:
            k = cluster_assignment.max()

        cluster_ids = []

        for cluster_no in xrange(k):
            indices, = np.where(cluster_assignment == cluster_no)
            size = len(indices)
            if size < min_size:
                continue

            pur, cat = self.cluster_purity(indices)
            if pur >= threshold:
                desc = { 'cluster': cluster_no, 'purity': pur, 'category': cat, 'size': size }
                if all_categories: 
                    desc['all_categories'] = self.cluster_categories(indices)
                cluster_ids.append(desc)

        return cluster_ids

    def combine_def(self, definition_list):
        key = lambda t: t[0].lower()
        sorted_by_def = sorted(definition_list, key=key)

        combined = []

        for definition, group in groupby(sorted_by_def, key=key):
            group = list(group)
            namespace_frequency = len(group)
            avg_score = np.mean([score for _, score in group])

            score = avg_score * namespace_frequency
            combined.append((definition, score))

        return sorted(combined, key=operator.itemgetter(1), reverse=True)

    def fuzzy_combine_def(self, definitions, scorer=None):
        d = dict(definitions) 
        order_key = lambda name: d[name]

        result = []        
        names = set(d.keys())

        if scorer is None:
            scorer = fuzz.token_set_ratio

        while names:
            first = names.pop()

            similar = process.extractBests(first, names, scorer=scorer, limit=1000, score_cutoff=65)
            similar_names = [name for name, _ in similar]

            for name in similar_names:
                names.remove(name)

            res_names = [first] + similar_names
            ordered_by_score = sorted(res_names, key=order_key, reverse=True)

            total_score = sum([d[name] for name in ordered_by_score])
            result.append((ordered_by_score, total_score))

        resorted = sorted(result, key=operator.itemgetter(1), reverse=True)
        return resorted

    def find_all_def(self, cluster_assignment, cluster_ids, scorer=None):
        if isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]
   
        all_relations = defaultdict(list)

        for cluster_id in cluster_ids:
            indices, = np.where(cluster_assignment == cluster_id)

            for np_idx in indices:
                idx = int(np_idx)
                for ident, definit in self.doc_ids_definitions[idx].items():
                    all_relations[ident].extend(definit)

        combined = {}
        for id, definitions in all_relations.items():
            pre_combined = self.combine_def(definitions)
            combined[id] = self.fuzzy_combine_def(pre_combined, scorer)

        return combined 

    def __fuzzy_list_name(self, list_names):
        if len(list_names) == 1:
            return list_names[0]
        else:
            return list_names[0] + '*'

    def _string_def_list(self, ident, def_list_fuzzy_combined):
        def_str = ['(%s: %0.2f)' % (self.__fuzzy_list_name(d), s) for (d, s) in def_list_fuzzy_combined]
        return '%s: %s' % (ident, ', '.join(def_str))

    def cluster_details(self, cluster_assignment, cluster_id):
        indices, = np.where(cluster_assignment == cluster_id)

        document_titles = []
        document_categories = Counter()

        for np_idx in indices:
            idx = int(np_idx)

            document_titles.append(self.doc_titles[idx])
            document_categories.update(self.doc_categories[idx])

        return document_titles, document_categories
