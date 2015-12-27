# -*- coding: utf-8 -*-

'''
Created on Oct 18, 2015

@author: alexey
'''

# IDENTIFIER VECTOR SPACE
# 5 ways: nodef, weak, strong, full, strong last

IDV_NO_DEF = 'nodef'
IDV_WEAK = 'weak'
IDV_STRONG = 'strong'
IDV_FULL = 'full'
IDV_STRONG_LAST = 'strong_last'

from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')

import logging
log = logging.getLogger('nd.algorithms')


def process_nodef(inp, stemmer):
    return inp

def process_weak(inp, stemmer):
    rels = inp.document_relations
    ids = inp.document_identifiers
    N_doc = len(rels)

    for idx in xrange(N_doc):
        vals = rels[idx].items()
        id_list = ids[idx]

        for id, definitions in vals:
            for definition, score in definitions:
                unigrams = definition.lower().split()
                id_list[id] = id_list[id] + 1

                for unigram in unigrams:
                    stem = stemmer(unigram)
                    id_list[stem] = id_list[stem] + 1

    return inp

def process_strong(inp, stemmer):
    rels = inp.document_relations
    ids = inp.document_identifiers
    N_doc = len(rels)

    for idx in xrange(N_doc):
        vals = rels[idx].items()
        id_list = ids[idx]

        for id, definitions in vals:
            for definition, score in definitions:
                for unigram in definition.lower().split():
                    stem = stemmer(unigram)
                    key = u'%s_%s' % (id, stem)
                    id_list[key] = id_list[key] + 1
    return inp


def process_full(inp, stemmer):
    rels = inp.document_relations
    ids = inp.document_identifiers
    N_doc = len(rels)

    for idx in xrange(N_doc):
        vals = rels[idx].items()
        id_list = ids[idx]

        for id, definitions in vals:
            for definition, score in definitions:
                normalized_def = []

                for unigram in definition.lower().split():
                    stem = stemmer(unigram)
                    normalized_def.append(stem)

                normalized_def = ' '.join(normalized_def)
                key = u'%s_%s' % (id, normalized_def)
                id_list[key] = id_list[key] + 1
    return inp


def process_strong_last(inp, stemmer):
    rels = inp.document_relations
    ids = inp.document_identifiers
    N_doc = len(rels)

    for idx in xrange(N_doc):
        vals = rels[idx].items()
        id_list = ids[idx]

        for id, definitions in vals:
            for definition, score in definitions:
                def_tokens = definition.lower().split()
                last = stemmer(def_tokens[-1])
                key = u'%s_%s' % (id, last)
                id_list[key] = id_list[key] + 1
    return inp


type_function_dict = {
    IDV_NO_DEF: process_nodef,
    IDV_WEAK: process_weak,
    IDV_STRONG: process_strong,
    IDV_FULL: process_full,
    IDV_STRONG_LAST: process_strong_last
}

def identity_stemmer(term):
    return term

type_stemmer_dict = {
    'snowball': snowball_stemmer,
    'none': identity_stemmer
}

def process(type, stemmer, inp):
    if type in type_function_dict:
        log.debug('using %s type of IVS' % type)

        if stemmer in type_stemmer_dict:
            stemmer_func = type_stemmer_dict[stemmer]
        else: 
            log.info('unknown stemmer type %s, using identity stemmer' % stemmer)
            stemmer_func = type_stemmer_dict['none']

        return type_function_dict[type](inp, stemmer_func)
    else:
        raise Exception('unknown type "%s"' % type)
    

from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorized():
    original_input = None
    vectorizer = None
    X = None

    def __init__(self, data, vectorizer, X):
        self.original_input = data
        self.vectorizer = vectorizer
        self.X = X

def unwrap_counter(cnt):
    res = []
    for id, c in cnt.items():
        res.extend([id] * c)
    return res

def vectorize(inp, **kwargs):
    """ Vectorizes the input 

    inp: input from nd.read.mlp_read.InputData
    kwargs: other params passed to sklearn vectorizer 
            such as use_idf=True, sublinear_tf=True 
    """

    log.debug('Vectorizing input using params %s' % str(kwargs))

    vectorizer = TfidfVectorizer(analyzer=unwrap_counter, min_df=2, **kwargs)

    ids = inp.document_identifiers
    X = vectorizer.fit_transform(ids)

    log.debug('Done. Resulted matrix is %d x %d' % X.shape)
    return Vectorized(data=inp, vectorizer=vectorizer, X=X)

