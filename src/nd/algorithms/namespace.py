'''
Created on Nov 8, 2015

@author: alexey
'''

from collections import defaultdict

import numpy as np

from nltk.tokenize import word_tokenize

from nltk.stem import SnowballStemmer

snowball_stemmer = SnowballStemmer('english')

from nltk.corpus import stopwords
ENGLISH_STOP_WORDS = set(stopwords.words('english') + 
                 ['etc', 'given', 'method', 'methods', 'theory', 'problem',
                  'problems', 'model', 'models'] + 
                 ['section'] + ['must', 'also'])

from sklearn.feature_extraction.text import TfidfVectorizer


class Namespace():

    _parent = None
    _name = None
    _identifiers = None
    _children = None
    _relations = None
    _wiki_cats = None

    _cluster_id = None

    def __init__(self, name, parent=None):
        self._name = name
        self._children = []

        if parent:
            self._parent = parent
            self._parent._children.append(self)

    def set_wiki_categories(self, wiki_cats):
        self._wiki_cats = wiki_cats

    def most_common_wiki_cat(self):
        if self._wiki_cats:
            return self._wiki_cats.most_common(1)[0]
        else:
            return None

    def set_relations(self, relations):
        self._relations = relations

    def set_additional_info(self, cluster_id, purity, matching_score, matching_terms):
        self._cluster_id = cluster_id
        self._purity = purity
        self._matching_score = matching_score
        self._matching_terms = matching_terms

    def to_dict(self, evaluator):
        res = {}
        res['category_name'] = self._name

        if self._wiki_cats:
            res['wiki_categories'] = self._wiki_cats.most_common(3)

        if self._cluster_id:
            res['cluster_id'] = self._cluster_id
            res['matching_score'] = self._matching_score
            res['purity'] = self._purity
            res['matching_keywords'] = list(self._matching_terms)

        if self._relations:
            relations = []
            for identifier, def_list in self._relations:
                definitions, score = def_list[0]
                top_definition = definitions[0] 
                relation = {'identifier': identifier, 
                            'top_definition': top_definition, 
                            'top_definition_score': score, 
                            'all_definitions': definitions }
                relations.append(relation)
            res['relations'] = relations

        children = []
        for child in self._children:
            child_dict = child.to_dict(evaluator)
            children.append(child_dict)
        res['children'] = children

        return res

    def print_ns(self, evaluator, indend=0, print_rels=0):
        indend_str = ' ' * (4 * indend)
        print indend_str, 'Category: %s' % self._name
        if self._wiki_cats:
            print indend_str, '          wiki categories:', 
            print ', '.join('%s (%d)' % (cat, cnt) for cat, cnt in self._wiki_cats.most_common(3))
        if self._cluster_id:
            print indend_str, '          cluster_id: %d (matching score: %0.2f, purity: %0.2f)' % \
                                    (self._cluster_id, self._matching_score, self._purity)
            print indend_str, '          common:', ' '.join(self._matching_terms)

        if print_rels and self._relations:
            for id, def_list in self._relations:
                print indend_str, '-',
                print evaluator._string_def_list(id, def_list) 

        print 
        for child in self._children:
            child.print_ns(evaluator, indend+1, print_rels=print_rels)
            print

    def __repr__(self):
        return self._name

def scheme_to_vsm(scheme):
    all_categories = []
    all_categories_idx = {}

    cnt = 0

    for k_top, top in scheme.items():
        for k_2, v in top.items():
            top = ' '.join([k_top] * 3)
            document = top + ' ' + k_2 + ' ' + ' '.join(v)
            tokens = word_tokenize(document)
            tokens = [t.lower() for t in tokens if t.isalpha()]
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
            all_categories.append(tokens)
            all_categories_idx[cnt] = (k_top, k_2)
            cnt = cnt + 1

    def identity(lst): return lst
    category_vectorizer = TfidfVectorizer(analyzer=identity).fit(all_categories)

    cat_index = category_vectorizer.transform(all_categories)
    return cat_index, category_vectorizer, all_categories, all_categories_idx


def clusters_to_vsm(labels, selected_clusters, mlp_data, evaluator, 
                    category_vectorizer):
    # selected_clusters = evaluate.high_purity_clusters(cluster_assignment, threshold=0.8, all_categories=1, min_size=3)
    desc_ids = [d['cluster'] for d in selected_clusters]
    rels = mlp_data.document_relations

    def counter_to_string(cnt, repeat=1):
        if repeat:
            return ' '.join([(word + ' ') * cnt for word, cnt in cl_cats.items()])
        else:
            return ' '.join(cnt.keys())

    def all_definitions(clustering, cluster_index):
        indices, = np.where(clustering == cluster_index)

        all_defs = []
        for idx in indices: 
            idx = int(idx)
            for lst in rels[idx].values():
                for d, _ in lst:
                    all_defs.extend(d.split())
        return all_defs

    clusters_representation = []

    for cl_id in desc_ids:
        cl_titles, cl_cats = evaluator.cluster_details(labels, cl_id)

        document = ' '.join(cl_titles) + ' ' + counter_to_string(cl_cats)
        tokens = word_tokenize(document)
        tokens = [t.lower() for t in tokens if t.isalpha()]
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

        clusters_representation.append(tokens)

    clus_index = category_vectorizer.transform(clusters_representation)
    return clus_index, clusters_representation


def assign_clusters_to_scheme(scheme, labels, mlp_data, evaluator, selected_clusters):
    scheme_vec, category_vectorizer, all_categories, all_categories_idx = scheme_to_vsm(scheme)
    clusters_vec, clusters_representation = clusters_to_vsm(labels, selected_clusters, 
                                                            mlp_data, evaluator, category_vectorizer)

    # cosine between scheme and clusters
    clus_cat_sim = (clusters_vec * scheme_vec.T).toarray()

    best_similarity = clus_cat_sim.max(axis=1)
    clus_cat_assignment = clus_cat_sim.argmax(axis=1)

    desc_ids = [d['cluster'] for d in selected_clusters]
    namespaces = defaultdict(list)
    namespace_name = []

    for idx in range(len(desc_ids)):
        cat_id = clus_cat_assignment[idx]
        desc = selected_clusters[idx]

        score = best_similarity[idx]
        common_keywords = set(clusters_representation[idx]) & set(all_categories[cat_id])
    
        parent_cat, namespace_cat = all_categories_idx[cat_id]
        if score <= 0.2 or len(common_keywords) == 1:
            parent_cat = 'OTHER'
    
        namespaces[parent_cat].append((namespace_cat, score, desc, common_keywords))
        namespace_name.append((desc['cluster'], (parent_cat, namespace_cat)))

    namespaces = sorted(namespaces.items())

    ROOT = Namespace('ROOT')

    for parent_cat, groups in namespaces:
        parent_namespace = Namespace(parent_cat, ROOT)
    
        for cat, score, desc, common in groups:
            ns = Namespace(cat, parent_namespace)
            ns.set_wiki_categories(desc['all_categories'])

            # wiki_category = ns.most_common_wiki_cat()
            cluster_id = desc['cluster']
            ns.set_additional_info(cluster_id, desc['purity'], score, common)
    
            all_def = evaluator.find_all_def(labels, cluster_id) 
            all_items = sorted(all_def.items())
            ns.set_relations(all_items)

    return ROOT




