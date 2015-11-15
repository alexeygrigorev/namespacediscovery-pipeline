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


class Evaluator:
    doc_titles = None
    doc_ids = None
    doc_ids_definitions = None
    doc_categories = None

    
    def _def_docfreq_calculate(self, doc_ids_definitions):
        docfreq_definitions = Counter()

        for rel_set in doc_ids_definitions:
            for _, def_list in rel_set.items():
                docfreq_definitions.update(set(d for d, s in def_list))


    def _category_docfreq_calculate(self, doc_categories):
        docfreq_categories = Counter()
        
        # doc_categories is a list of sets with categories
        for categories in doc_categories:
            docfreq_categories.update(categories)

        return {d: np.log(s + 1) for d, s in docfreq_categories.items()}


    def __init__(self, doc_titles, doc_ids, doc_ids_definitions, doc_categories):
        self.doc_titles = doc_titles
        self.doc_ids = doc_ids
        self.doc_ids_definitions = doc_ids_definitions
        self.doc_categories = doc_categories
        self.doc_df_categories = self._category_docfreq_calculate(doc_categories)

        self._def_docfreq_calculate(doc_ids_definitions)

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


    def overall_purity(self, cluster_assignment, k=None):
        if not k:
            k = cluster_assignment.max()

        purities = []
        sizes = []

        for cluster_no in xrange(k):
            pur, _ = self.cluster_purity(cluster_assignment == cluster_no)
            purities.append(pur)
            sizes.append(np.sum(cluster_assignment == cluster_no))

        purity = np.array(purities)
        sizes = np.array(sizes, dtype=np.float) / len(cluster_assignment)
        
        return purity.dot(sizes)


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


    def report_overall(self, cluster_assignment, purity_threshold, min_size=5, 
                       k=None, sort_by='purity'):
        pure_clusters = self.high_purity_clusters(cluster_assignment, purity_threshold, min_size, k)
        
        print 'overall purity %0.4f' % self.overall_purity(cluster_assignment)
        print 'number of high purity clusters of size at least %d is %d' % \
                         (min_size, len(pure_clusters))
        print

        pure_clusters = sorted(pure_clusters, key=operator.itemgetter(sort_by), reverse=True)

        for desc in pure_clusters: 
            print '- %s (id=%d) size=%d, purity=%0.4f' % \
                    (desc['category'], desc['cluster'], desc['size'], desc['purity'])


    def find_identifier(self, cluster_assignment, purity_threshold, id, 
                        min_size=5, collection_weighting=0, scorer=None):
        pure_clusters = self.high_purity_clusters(cluster_assignment, 
                                                  purity_threshold, min_size, k=None)

        print 'overall purity %0.4f' % self.overall_purity(cluster_assignment)
        print 'number of high purity clusters of size at least 5 is %d' % len(pure_clusters)
        print

        for desc in pure_clusters: 
            cluster_id = desc['cluster']

            indices, = np.where(cluster_assignment == cluster_id)
            size = len(indices)

            occurrences = []
            cluster_cats = Counter()

            for np_idx in indices:
                idx = int(np_idx)
                cluster_cats.update(self.doc_categories[idx])

                if id not in self.doc_ids_definitions[idx]:
                    continue

                definition_list = self.doc_ids_definitions[idx][id]
                if definition_list:
                    occurrences.extend(definition_list)

            if not occurrences:
                continue

            top_categories = cluster_cats.most_common()[:5]
            print 'category "%s", cluster_id=%d, size=%d:' % (top_categories[0][0], cluster_id, size)
            print 'top categories:', top_categories

            print '    ',
            print self.print_fuzzy_merged_definition_list(id, occurrences, scorer=scorer)


    def combine_def(self, definition_list, collection_weighting=0):
        key = lambda t: t[0].lower()
        sorted_by_def = sorted(definition_list, key=key)

        combined = []

        for definition, group in groupby(sorted_by_def, key=key):
            group = list(group)
            namespace_frequency = len(group)
            avg_score = np.mean([score for _, score in group])

            score = avg_score * namespace_frequency
            if collection_weighting:
                idf = self.doc_idf_definitions[definition]
                score = score * idf

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
   
        all_rels = defaultdict(list)

        for cluster_id in cluster_ids:
            indices, = np.where(cluster_assignment == cluster_id)

            for np_idx in indices:
                idx = int(np_idx)
                for ident, definit in self.doc_ids_definitions[idx].items():
                    all_rels[ident].extend(definit)

        combined = {}
        for id, defininitions in all_rels.items():
            pre_combined = self.combine_def(defininitions)
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

    def print_fuzzy_merged_definition_list(self, ident, def_list, collection_weighting=0, scorer=None):
        pre_combined = self.combine_def(def_list, collection_weighting=collection_weighting)
        combined = self.fuzzy_combine_def(pre_combined, scorer=scorer)
        return self._string_def_list(ident, combined)


    def print_cluster(self, cluster_assignment, cluster_ids, collection_weighting=0, 
                      scorer=None, print_docs=1, sort_by_score=0,
                      normalize_score=0, top_k_def=None, sort_docs=0):
        if isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]

        all_rels = defaultdict(list)
        all_sets = []
        cluster_cats = Counter()

        for cluster_id in cluster_ids:
            indices, = np.where(cluster_assignment == cluster_id)

            size = len(indices)
            if print_docs:
                print 'cluster %d, size: %d' % (cluster_id, size)

            for np_idx in indices:
                idx = int(np_idx)

                all_sets.append(set(self.doc_ids[idx]))
                cluster_cats.update(self.doc_categories[idx])

                if print_docs:
                    print '-', self.doc_titles[idx],

                    categories = list(self.doc_categories[idx])
                    categories = sorted(categories, key=lambda k: self.doc_df_categories[k], reverse=True)
                    if len(categories) > 5:
                        categories = categories[:5] + ['...']

                    print '(categories: %s)' % ', '.join(categories),
                    print ' '.join(self.doc_ids[idx])

                for ident, definit in self.doc_ids_definitions[idx].items():
                    all_rels[ident].extend(definit)
            if print_docs:
                print

        common_terms = set.intersection(*all_sets)
        print 'common terms: (%s)' % ' '.join(common_terms)
        print 'top categories:', ', '.join(u'(%s, %d)' % (n, c) for n, c in cluster_cats.most_common()[:5])

        most_common_cat, most_common_cat_count = cluster_cats.most_common()[0]
        print 'purity: %0.3f' % (1.0 * most_common_cat_count / size)

        print 'relations:'

        all_rels = sorted(all_rels.items(), key=operator.itemgetter(0))

        if sort_by_score:
            all_defs = []

            for ident, definit_list in all_rels:
                pre_combined = self.combine_def(definit_list, collection_weighting=collection_weighting)
                best = self.fuzzy_combine_def(pre_combined, scorer=scorer)[0]
                all_defs.append((ident, self.__fuzzy_list_name(best[0]), best[1]))

            all_defs = sorted(all_defs, key=operator.itemgetter(2), reverse=True)

            if top_k_def is not None:
                all_defs = all_defs[:top_k_def]

            for ident, definition, score in all_defs:
                if normalize_score:
                    score = np.tanh(score / 2.0)
                print '%s: %s (%.2f)' % (ident, definition, score)
        else:
            for ident, definit_list in all_rels:
                print '    ', 
                print self.print_fuzzy_merged_definition_list(ident, definit_list, scorer=scorer,
                                                              collection_weighting=collection_weighting)

    def cluster_details(self, cluster_assignment, cluster_id):
        indices, = np.where(cluster_assignment == cluster_id)

        document_titles = []
        document_categories = Counter()

        for np_idx in indices:
            idx = int(np_idx)

            document_titles.append(self.doc_titles[idx])
            document_categories.update(self.doc_categories[idx])

        return document_titles, document_categories

    def describe(self, cluster_assignment, top_categories=4):
        k = cluster_assignment.max()

        for k_i in xrange(k):
            indices, = np.where(cluster_assignment == k_i)
            N_i = len(indices) * 1.0

            cluster_cats = Counter()

            for doc_i in indices:
                idx = int(doc_i)
                cluster_cats.update(self.doc_categories[idx])

            relative = [(cat, freq / N_i) for cat, freq in cluster_cats.most_common(top_categories)]
            rel = ['%s (%0.4f)' % (c, f) for c, f in relative]
            print k_i, ','.join(rel)

    def category_distribution(self, cluster_assignment, category, order=None, show_zero=0):
        k = cluster_assignment.max()

        category_involvement = []

        for k_i in xrange(k):
            indices, = np.where(cluster_assignment == k_i)
            N_i = len(indices)

            cluster_cats = Counter()

            for doc_i in indices:
                idx = int(doc_i)
                cluster_cats.update(self.doc_categories[idx])
 
            count = cluster_cats[category]
            if not show_zero and count == 0:
                continue

            category_involvement.append((k_i, count, N_i))

        total = sum(cnt for k_i, cnt, N_i in category_involvement)
        category_involvement = \
            [(k_i, cnt, 100.0 * cnt / total, N_i, 1.0 * cnt / N_i) for (k_i, cnt, N_i) in category_involvement]

        if order:
            order_dict = {'percentage': 2, 'count': 1, 'size': 3, 'purity': 4}
            key = operator.itemgetter(order_dict[order])
            category_involvement = sorted(category_involvement, key=key, reverse=True)

        for k_i, count, percentage, N_i, purity in category_involvement:
            print '%3d: percentage: %2.1f%%, coverage: %d/%d = %0.4f'  % (k_i, percentage, count, N_i, purity)

