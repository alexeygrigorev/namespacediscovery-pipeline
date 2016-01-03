# -*- coding: utf-8 -*-

'''
Created on Oct 18, 2015

@author: alexey
'''

import json
import os

from collections import Counter
from collections import defaultdict

import logging

log = logging.getLogger('nd.read')


def id_counter(id_list):
    """ Converts a list of json objects from MLP into a Counter of 
    identifiers 
    """
    cnt = Counter()
    for el in id_list:
        cnt[el[u'element']] = el[u'count']

    return cnt


def_black_list = {
    'unit', 'units', 'value', 'values', 'axis', 'axes', 'factor', 'factors', 'line', 'lines',
    'point', 'points', 'number', 'numbers', 'variable', 'variables', 'respect', 'case', 'cases',
    'vector', 'vectors', 'element', 'elements', 'example',
    'integer', 'integers', 'term', 'terms', 'parameter', 'parameters', 'coefficient', 'coefficients',
    'formula', 'times', 'product', 'matrices', 'expression', 'complex', 'real', 'zeros', 'bits',
    'sign',
    'if and only if',
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda',
    'mu', 'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega'
}


def valid_def(definition):
    """ Returns true if passed definition is considered valid.
    That is, it's not too short and it's not in the black list
    
    """
    if len(definition) <= 3:
        return False

    return definition.lower() not in def_black_list


def rel_to_dict(rels):
    """ Converts a list of json objects from MLP into a Dict of 
    ID -> [(definition, score)].
    
    That is, each identifier is mapped to a list of (definition, score)
    tuples
    """

    res = defaultdict(list)
    for r in rels:
        if not valid_def(r['definition']):
            continue
        res[r['identifier']].append((r['definition'], r['score']))
    return res


def read_categories(path):
    """ reads category information from a file and returns 
    two dictionaries: 
    
    1. doc -> [categories] (maps each document to a list of categories)
    2. category -> [docs] (maps each category to a list of documents)
    
    """

    doc_categories = defaultdict(set)
    category_docs = defaultdict(set)

    for line in file(path):
        title, cat = line.strip().split('\t')
        title = title.decode('utf-8')
        cat = cat.decode('utf-8')

        # let's also remove all documents from "OTHER" category
        if cat == u'OTHER':
            continue

        doc_categories[title].add(cat)
        category_docs[cat].add(title)

    return doc_categories, category_docs


def read_mlp_output(mlp_output_dir, doc_categories):
    log.debug('reading MLP output from %s...' % mlp_output_dir)
    docs = []
    titles = []
    ids = []
    rels = []

    empty = 0
    small = 0
    uncategorized = 0

    for f in os.listdir(mlp_output_dir):
        for line in file(mlp_output_dir + f):
            doc = json.loads(line)

            title = doc['title']
            if title not in doc_categories:
                uncategorized = uncategorized + 1
                continue

            if '(disambiguation)' in title:
                continue

            id_bag = id_counter(doc['identifiers'])
            if len(id_bag) <= 1:
                if len(id_bag) == 0:
                    empty = empty + 1
                else:
                    small = small + 1
                continue

            docs.append(doc)
            titles.append(title)
            ids.append(id_bag)

            id_rels = rel_to_dict(doc['relations'])
            rels.append(id_rels)

    log.debug('skipped %d empty, %d small and %d uncategorized documents' %
              (empty, small, uncategorized))

    N_doc = len(ids)
    log.debug('read %d documents' % N_doc)
    return titles, ids, rels


def build_doc_category_list(doc_categories, category_docs, titles, title_idx):
    # processing category information 
    # so we don't keep category information for documents we don't have
    for doc, cats in doc_categories.items():
        if doc in title_idx:
            continue

        for cat in cats:
            category_docs[cat].remove(doc)

        del doc_categories[doc]

    return [doc_categories[doc] for doc in titles]


def main_read(props):
    """ Reads the MLP output from props['input'] and category information
    from props['categories'], returns an InputData object with
    titles, categories, identifiers, relations
    """

    categories_file = props['categories']
    doc_categories, category_docs = read_categories(categories_file)

    mlp_output_dir = props['input']
    titles, ids, rels = read_mlp_output(mlp_output_dir, doc_categories)

    # document to index dictionary
    title_idx = {title: idx for (idx, title) in enumerate(titles)}

    doc_categories_list = build_doc_category_list(doc_categories, category_docs, titles, title_idx)

    result = InputData(document_titles=titles,
                       document_categories=doc_categories_list,
                       document_identifiers=ids,
                       document_relations=rels,
                       document_to_index=title_idx)

    result.remove_infrequent_definitions(threshold=1)
    result.remove_infrequent_identifiers(threshold=2)

    return result


class InputData():
    def __init__(self, document_titles, document_categories,
                 document_identifiers, document_relations, document_to_index):
        """
        :rtype: InputData
        """
        self.document_titles = document_titles
        self.document_categories = document_categories
        self.document_identifiers = document_identifiers
        self.document_relations = document_relations
        self.document_to_index = document_to_index

    def remove_infrequent_definitions(self, threshold=1):
        log.debug('removing infrequent definitions with frequency <= %d...' % threshold)
        def_freq = Counter()

        for def_dict in self.document_relations:
            for _, def_list in def_dict.items():
                def_freq.update([d for d, _ in def_list])

        low_freq_def = {id for id, cnt in def_freq.items() if cnt <= threshold}
        log.debug('there are %d definitions with frequency <= %d, removing them...' %
                  (len(low_freq_def), threshold))

        for def_dict in self.document_relations:
            for id, def_list in def_dict.items():
                clean_def_list = []
                for definition, score in def_list:
                    if definition not in low_freq_def:
                        clean_def_list.append((definition, score))

                if not clean_def_list:
                    del def_dict[id]
                else:
                    def_dict[id] = clean_def_list

        log.debug('removed')

    def remove_infrequent_identifiers(self, threshold=2):
        log.debug('removing infrequent identifiers with frequency <= %d...' % threshold)
        all_ids = Counter()
        for id_cnt in self.document_identifiers:
            all_ids.update(id_cnt)

        infrequent = set()

        for (el, cnt) in all_ids.items():
            if cnt <= threshold:
                infrequent.add(el)

        for id_cnt in self.document_identifiers:
            for id in (set(id_cnt) & infrequent):
                del id_cnt[id]

        log.debug('removed')


if __name__ == '__main__':
    pass
