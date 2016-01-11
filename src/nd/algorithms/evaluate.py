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


class ClusterDescription():
    def __init__(self, cluster, purity, category, size, all_categories):
        self.cluster = cluster
        self.purity = purity
        self.category = category
        self.size = size
        self.all_categories = all_categories


class Evaluator():
    def __init__(self, document_titles, document_identifiers, document_relations, document_categories):
        self.document_titles = document_titles
        self.document_identifiers = document_identifiers
        self.document_relations = document_relations
        self.document_categories = document_categories

    def cluster_purity(self, indices):
        """
        Calculates purity of a cluster
        :param indices: indexes of documents for which we want to calculate purity
        :return: purity and most frequent category
        :rtype: tuple(float, str)
        """
        if indices.dtype == 'bool':
            indices, = np.where(indices)

        size = len(indices)
        if size == 0:
            return 0.0, ''

        cluster_cats = Counter()

        for idx in indices:
            cluster_cats.update(self.document_categories[idx])

        if len(cluster_cats) == 0:
            return 0.0, ''

        category, cat_cnt = cluster_cats.most_common()[0]
        return 1.0 * cat_cnt / size, category

    def cluster_categories(self, indices):
        """

        :param indices: indexes of documents for which we want to calculate purity
        :return: categories for the cluster
        :rtype: Counter
        """
        if indices.dtype == 'bool':
            indices, = np.where(indices)

        cluster_cats = Counter()
        for idx in indices:
            cluster_cats.update(self.document_categories[idx])

        return cluster_cats

    def high_purity_clusters(self, cluster_assignment, threshold, min_size=5):
        k = cluster_assignment.max()

        cluster_ids = []

        for cluster_no in xrange(k):
            indices, = np.where(cluster_assignment == cluster_no)
            size = len(indices)
            if size < min_size:
                continue

            purity, category = self.cluster_purity(indices)
            if purity >= threshold:
                desc = ClusterDescription(cluster=cluster_no, purity=purity,
                                          category=category, size=size,
                                          all_categories=self.cluster_categories(indices))
                cluster_ids.append(desc)

        return cluster_ids

    def cluster_description(self, cluster_assignment, cluster_id):
        indices, = np.where(cluster_assignment == cluster_id)
        size = len(indices)

        purity, category = self.cluster_purity(indices)
        desc = ClusterDescription(cluster=cluster_id, purity=purity,
                                  category=category, size=size,
                                  all_categories=self.cluster_categories(indices))
        return desc


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
        if isinstance(cluster_ids, (int, np.int32)):
            cluster_ids = [cluster_ids]

        all_relations = defaultdict(list)

        for cluster_id in cluster_ids:
            indices, = np.where(cluster_assignment == cluster_id)

            for np_idx in indices:
                idx = int(np_idx)
                for identifier, definition in self.document_relations[idx].items():
                    all_relations[identifier].extend(definition)

        combined = {}
        for id, definitions in all_relations.items():
            pre_combined = self.combine_def(definitions)
            combined[id] = self.fuzzy_combine_def(pre_combined, scorer)

        return combined

    def document_definitions(self, document_id, scorer=None):
        all_relations = defaultdict(list)
        for identifier, definition in self.document_relations[document_id].items():
            all_relations[identifier].extend(definition)

        combined = {}
        for id, definitions in all_relations.items():
            pre_combined = self.combine_def(definitions)
            combined[id] = self.fuzzy_combine_def(pre_combined, scorer)

        return combined


    # UNUSED!
    def __fuzzy_list_name(self, list_names):
        if len(list_names) == 1:
            return list_names[0]
        else:
            return list_names[0] + '*'

    # UNUSED!
    def _string_def_list(self, indent, def_list_fuzzy_combined):
        def_str = ['(%s: %0.2f)' % (self.__fuzzy_list_name(d), s) for (d, s) in def_list_fuzzy_combined]
        return '%s: %s' % (indent, ', '.join(def_str))

    def cluster_details(self, cluster_assignment, cluster_id):
        indices, = np.where(cluster_assignment == cluster_id)

        document_titles = []
        document_categories = Counter()

        for np_idx in indices:
            idx = int(np_idx)

            document_titles.append(self.document_titles[idx])
            document_categories.update(self.document_categories[idx])

        return document_titles, document_categories
