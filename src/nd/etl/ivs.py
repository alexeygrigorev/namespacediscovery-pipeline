# -*- coding: utf-8 -*-

'''
Created on Oct 18, 2015

@author: alexey
'''

# IDENTIFIER VECTOR SPACE
# 3 ways: nodef, weak, strong

IDV_NO_DEF = 'nodef'
IDV_WEAK = 'weak'
IDV_STRONG = 'strong'

from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')

import logging
log = logging.getLogger('nd.etl.ivs')

def process_nodef(inp):
    return inp

def process_weak(inp):
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
                    stem = snowball_stemmer.stem(unigram)
                    id_list[stem] = id_list[stem] + 1

    return inp

def process_strong(inp):
    return inp


type_function_dict = {
    IDV_NO_DEF: process_nodef,
    IDV_WEAK: process_weak,
    IDV_STRONG: process_strong 
}

def process(type, inp):
    if type in type_function_dict:
        log.debug('using %s type of IVS' % type)
        return type_function_dict[type](inp)
    else:
        raise Exception('unknown type "%s"' % type)
    


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, randomized_svd


class Vectorized():
    original_input = None
    vectorizer = None
    X = None

    def __init__(self, data, vectorizer, X):
        self.original_input = data
        self.vectorizer = vectorizer
        self.X = X

def vectorize(inp, **kwargs):
    """ Vectorizes the input 

    inp: input from nd.etl.read.InputData
    kwargs: other params passed to sklearn vectorizer 
            such as use_idf=True, sublinear_tf=True 
    """

    def unwrap_counter(cnt):
        res = []
        for id, c in cnt.items():
            res.extend([id] * c)
        return res

    log.debug('Vectorizing input using params %s' % str(kwargs))

    vectorizer = TfidfVectorizer(analyzer=unwrap_counter, min_df=2, **kwargs)

    ids = inp.document_identifiers
    X = vectorizer.fit_transform(ids)

    log.debug('Done. Resulted matrix is %d x %d' % X.shape)
    return Vectorized(data=inp, vectorizer=vectorizer, X=X)

