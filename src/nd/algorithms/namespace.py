'''
Created on Nov 8, 2015

@author: alexey
'''

from collections import defaultdict
from collections import Counter

from itertools import groupby
import operator

import numpy as np

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from nd.read.scheme import ClassificationCategory

import logging
log = logging.getLogger('nd.algorithms')


ENGLISH_STOP_WORDS = set(stopwords.words('english') +
                         ['etc', 'given', 'method', 'methods', 'theory', 'problem',
                          'problems', 'model', 'models'] +
                         ['section'] + ['must', 'also'])
snowball_stemmer = SnowballStemmer('english')


class Namespace():
    def __init__(self, name, parent=None):
        """

        :rtype: Namespace
        """
        self._name = name
        self._children = []

        if parent:
            self._parent = parent
            self._parent._children.append(self)

        self._identifiers = None
        self._relations = None
        self._wiki_cats = None

        self._cluster_id = None
        self._cat_reference = None

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

        if self._cat_reference:
            res['reference_category'] = self._cat_reference.to_dict()

        if self._cluster_id:
            res['cluster_id'] = self._cluster_id
            res['matching_score'] = self._matching_score
            res['purity'] = self._purity
            res['matching_keywords'] = list(self._matching_terms)

        if self._relations:
            res['relations'] = combine_relations(self._relations)

        children = []
        for child in self._children:
            child_dict = child.to_dict(evaluator)
            children.append(child_dict)
        res['children'] = children

        return res

    def to_wiki(self, evaluator, f):
        dto = self.to_dict(evaluator)
        _print_dict(dto, evaluator, f, 1)

    def __repr__(self):
        return self._name

def is_link(definition):
    return definition.startswith('[[') and definition.endswith(']]')

def combine_relations(relations):
    combined = []
    for identifier, def_list in relations:
        definitions, score = def_list[0]
        links = [d for d in definitions if is_link(d)]

        if links:
            top_definition = links[0]
        else:
            top_definition = definitions[0]

        relation = {'identifier': identifier,
                    'top_definition': top_definition,
                    'top_definition_score': score,
                    'all_definitions': definitions}
        combined.append(relation)
    return combined


def _print_dict(res, evaluator, f, indent=1):
    capt = '=' * indent
    print >> f, capt, res['category_name'], capt

    if 'reference_category' in res:
        print >> f, 'Category information'
        category_info = res['reference_category']
        print >> f, '* reference category: ', category_info['full_name']
        print >> f, '* code: %s (%s)' % (category_info['code'], category_info['source'])

        if 'wiki_categories' in res:
            print >> f, '* wiki categories:', res['wiki_categories']
            print >> f
    else:
        if 'wiki_categories' in res:
            print >> f, 'Category information'
            print >> f, '* wiki categories:', res['wiki_categories']
            print >> f

    if 'cluster_id' in res:
        print >> f, 'Cluster information:'
        print >> f, '* cluster id:', res['cluster_id']
        print >> f, '* matching score:', res['matching_score']
        print >> f, '* purity:', res['purity']
        print >> f, '* matching keywords:', res['matching_keywords']
        print >> f

    if 'relations' in res:
        all_res = [(r['identifier'], r['top_definition'], r['top_definition_score'])
                   for r in res['relations']]
        all_res = sorted(all_res, key=operator.itemgetter(2), reverse=True)

        print >> f, "'''Definitions:'''"
        for id, definition, score in all_res:
            print >> f, u'* <math>%s</math>: %s (%0.3f)' % (id, definition, score)

    print >> f
    print >> f
    if 'children' in res:
        new_indent = indent + 1
        for child in res['children']:
            _print_dict(child, evaluator, f, new_indent)


class SchemeVSM():
    def __init__(self, scheme_vec, category_vectorizer, all_categories,
                 all_categories_idx, all_categories_meta):
        """
        :param scheme_vec: VSM built from terms of the scheme
        :param category_vectorizer: vectorizer used for building the VSM
        :param all_categories: tokens used for bulding the VSM
        :param all_categories_idx: index of scheme_vec to name of the category
        :param all_categories_meta: index of scheme_vec to meta information about category
        """
        self.scheme_vec = scheme_vec
        self.category_vectorizer = category_vectorizer
        self.all_categories = all_categories
        self.all_categories_idx = all_categories_idx
        self.all_categories_meta = all_categories_meta


class ClusterVSM():
    def __init__(self, clusters_vec, clusters_representation):
        """
        :param clusters_vec: clusters represented in the vector space (from SchemeVSM)
        :param clusters_representation: actual tokens from which the VSM was built
        """
        self.clusters_vec = clusters_vec
        self.clusters_representation = clusters_representation


def scheme_to_vsm(scheme):
    """
    Converts the reference schemes (MSC, PACS, ACM) to Vector Space Model

    :param scheme: reference scheme
    :type scheme: dict
    :returns: VSM representation of the scheme
    :rtype: SchemeVSM
    """

    all_categories = []
    all_categories_idx = {}
    all_categories_meta = {}
    cnt = 0

    for k_top, top in scheme.items():
        for k_2, values in top.items():
            top = ' '.join([k_top.full_name] * 3)
            cat_names = [c.full_name for c in values]
            document = top + ' ' + k_2.full_name + ' ' + ' '.join(cat_names)
            tokens = word_tokenize(document)
            tokens = [t.lower() for t in tokens if t.isalpha()]
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
            all_categories.append(tokens)
            all_categories_idx[cnt] = (k_top.full_name, k_2.full_name)
            all_categories_meta[cnt] = k_2
            cnt = cnt + 1

    def identity(lst):
        return lst

    category_vectorizer = TfidfVectorizer(analyzer=identity).fit(all_categories)

    cat_index = category_vectorizer.transform(all_categories)

    result = SchemeVSM(scheme_vec=cat_index,
                       category_vectorizer=category_vectorizer,
                       all_categories=all_categories,
                       all_categories_idx=all_categories_idx,
                       all_categories_meta=all_categories_meta)
    return result


def pure_clusters_to_vsm(labels, selected_clusters, mlp_data, evaluator,
                         category_vectorizer):
    """
    :param labels: cluster assignment labels as returned by sklearn
    :type labels:
    :param selected_clusters: clusters with high purity
    :type selected_clusters: list
    :type mlp_data: nd.read.mlp_read.InputData
    :type evaluator: nd.algorithms.evaluate.Evaluator
    :param category_vectorizer:
    :rtype: ClusterVSM
    """
    desc_ids = [d.cluster for d in selected_clusters]
    return buid_cluster_vsm(labels, desc_ids, evaluator, category_vectorizer)


def buid_cluster_vsm(labels, cluster_ids, evaluator, category_vectorizer):
    def counter_to_string(cnt, repeat=1):
        if repeat:
            return ' '.join([(word + ' ') * cnt for word, cnt in cl_cats.items()])
        else:
            return ' '.join(cnt.keys())

    clusters_representation = []
    for cl_id in cluster_ids:
        cl_titles, cl_cats = evaluator.cluster_details(labels, cl_id)

        document = ' '.join(cl_titles) + ' ' + counter_to_string(cl_cats)
        tokens = word_tokenize(document)
        tokens = [t.lower() for t in tokens if t.isalpha()]
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

        clusters_representation.append(tokens)
    cluster_vec = category_vectorizer.transform(clusters_representation)
    result = ClusterVSM(clusters_vec=cluster_vec,
                        clusters_representation=clusters_representation)
    return result


class NamespaceEntry():
    def __init__(self, namespace_cat, score, desc, common_keywords, cat_id):
        self.namespace_cat = namespace_cat
        self.score = score
        self.desc = desc
        self.common_keywords = common_keywords
        self.cat_id = cat_id
        self.referenced = False
        self.schema_meta = ClassificationCategory('NO_REFERENCE_CATEGORY', 'NO_CODE', 'NO_SCHEMA')

    def reference_category(self):
        if self.referenced:
            return self.schema_meta.code
        else:
            return self.desc.all_categories

def union(iterable):
    res = set()
    for s in iterable:
        res.update(s)
    return res


def union_counter(iterable):
    res = Counter()
    for s in iterable:
        res.update(s)
    return res

def assign_clusters_to_scheme(scheme, labels, mlp_data, evaluator, selected_clusters):
    """

    :param scheme:
    :param labels:
    :param mlp_data:
    :param evaluator:
    :param selected_clusters:
    :type selected_clusters: list of nd.algorithm.evaluate.ClusterDescription
    :return:
    """
    scheme_vsm = scheme_to_vsm(scheme)
    cluster_vsm = pure_clusters_to_vsm(labels, selected_clusters, mlp_data, evaluator,
                                       scheme_vsm.category_vectorizer)

    # cosine between scheme and clusters
    clus_cat_sim = (cluster_vsm.clusters_vec * scheme_vsm.scheme_vec.T).toarray()

    best_similarity = clus_cat_sim.max(axis=1)
    clus_cat_assignment = clus_cat_sim.argmax(axis=1)

    desc_ids = [d.cluster for d in selected_clusters]
    namespaces = defaultdict(list)
    namespace_name = []

    for idx in range(len(desc_ids)):
        cat_id = clus_cat_assignment[idx]
        desc = selected_clusters[idx]

        score = best_similarity[idx]
        common_keywords = set(cluster_vsm.clusters_representation[idx]) & \
                          set(scheme_vsm.all_categories[cat_id])

        parent_cat, namespace_cat = scheme_vsm.all_categories_idx[cat_id]
        entry = NamespaceEntry(namespace_cat, score, desc, common_keywords, cat_id)

        if score <= 0.2 or len(common_keywords) == 1:
            parent_cat = 'OTHER'
        else:
            entry.schema_meta = scheme_vsm.all_categories_meta[cat_id]
            entry.referenced = True

        namespaces[parent_cat].append(entry)
        namespace_name.append((desc.cluster, (parent_cat, namespace_cat)))

    namespaces = sorted(namespaces.items())

    ROOT = Namespace('ROOT')

    for parent_cat, groups in namespaces:
        parent_namespace = Namespace(parent_cat, ROOT)

        key = lambda e: e.reference_category()

        sorted_groups = sorted(groups, key=key)
        for k, values in groupby(sorted_groups, key=key):
            values = list(values)
            cat = values[0].namespace_cat
            score = np.mean([e.score for e in values])
            purity = np.mean([e.desc.purity for e in values])
            common = union(e.common_keywords for e in values)
            meta = values[0].schema_meta
            wiki_categories = union_counter(e.desc.all_categories for e in values)

            ns = Namespace(cat, parent_namespace)
            ns.set_wiki_categories(wiki_categories)
            ns._cat_reference = meta

            cluster_ids = [e.desc.cluster for e in values]
            ns.set_additional_info(cluster_ids, purity, score, common)

            all_def = evaluator.find_all_def(labels, cluster_ids)
            all_items = sorted(all_def.items())
            ns.set_relations(all_items)

    return ROOT


def namespaces_for_titles(scheme, labels, evaluator, mlp_data, doc_titles):
    """
    given a list of document titles, finds namespaces for them

    TODO: these two methods share a lot of code - refactor it

    :param scheme:
    :param labels:
    :param evaluator:
    :type mlp_data: nd.read.mlp_read.InputData
    :param doc_titles:
    :return:
    """
    scheme_vsm = scheme_to_vsm(scheme)

    cluster_ids = []
    doc_titles_selected = []

    for title in doc_titles:
        if title in mlp_data.document_to_index:
            doc_id = mlp_data.document_to_index[title]
            doc_titles_selected.append(title)
            cluster_ids.append(labels[doc_id])
        else:
            logging.info('cannot find index for document with title "%s", skipping it...' % title)

    cluster_vsm = buid_cluster_vsm(labels, cluster_ids, evaluator,
                                   scheme_vsm.category_vectorizer)

    clus_cat_sim = (cluster_vsm.clusters_vec * scheme_vsm.scheme_vec.T).toarray()
    best_similarity = clus_cat_sim.max(axis=1)
    clus_cat_assignment = clus_cat_sim.argmax(axis=1)

    result = []

    for i, cluster_id in enumerate(range(len(cluster_ids))):
        cat_id = clus_cat_assignment[cluster_id]
        desc = evaluator.cluster_description(labels, cluster_ids[cluster_id])

        score = best_similarity[cluster_id]
        common_keywords = set(cluster_vsm.clusters_representation[cluster_id]) & \
                          set(scheme_vsm.all_categories[cat_id])

        parent_cat, namespace_cat = scheme_vsm.all_categories_idx[cat_id]
        if score <= 0.2 or len(common_keywords) == 1:
            namespace_name = 'OTHER'
        else:
            namespace_name = parent_cat + ' -- ' + namespace_cat

        ns = Namespace(namespace_name, None)
        ns.set_wiki_categories(desc.all_categories)

        if namespace_name != 'OTHER':
            schema_meta = scheme_vsm.all_categories_meta[cat_id]
            ns._cat_reference = schema_meta
        else:
            no_ref = ClassificationCategory('NO_REFERENCE_CATEGORY', 'NO_CODE', 'NO_SCHEMA')
            ns._cat_reference = no_ref

        cluster_id = desc.cluster
        ns.set_additional_info(cluster_id, desc.purity, score, common_keywords)

        all_def = evaluator.find_all_def(labels, cluster_id)
        all_items = sorted(all_def.items())
        ns.set_relations(all_items)

        result_dict = ns.to_dict(evaluator)
        doc_title = doc_titles_selected[i]
        result_dict['document_title'] = doc_title
        result_dict['namespace_relations'] = result_dict['relations']
        del result_dict['relations']

        doc_id = mlp_data.document_to_index[doc_title]
        doc_relations = evaluator.document_definitions(doc_id)
        doc_relations = sorted(doc_relations.items())
        result_dict['document_relations'] = combine_relations(doc_relations)

        result_dict['other_documents'], _ = evaluator.cluster_details(labels, cluster_id)

        result.append(result_dict)

    return result
