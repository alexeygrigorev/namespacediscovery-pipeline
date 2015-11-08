'''
Created on Nov 8, 2015

@author: alexey
'''
from collections import defaultdict

import logging
log = logging.getLogger('nd.read')


def read_msc(path):
    tree = {}

    top_parent = None
    parent = None

    for line in file(path):
        if line.startswith('#'):
            continue

        tabs = sum(1 for s in line[:8] if s == ' ') / 4
        code, name = line.strip().split(': ', 1)
        name = name.decode('utf-8')
        name = name.replace('$', '')

        if code.endswith('99'):
            # 'Miscellaneous topics' or 'None of the above, but in this section'
            continue

        if tabs == 0:
            top_parent = name
            tree[top_parent] = {}
        elif tabs == 1:
            parent = name
            tree[top_parent][parent] = []
        else:
            tree[top_parent][parent].append(name)

    del tree['General']
    del tree['History and biography']
    del tree['Mathematics education']

    del tree['Quantum theory']['Axiomatics, foundations, philosophy']
    del tree['Quantum theory']['Applications to specific physical systems']
    del tree['Quantum theory']['Groups and algebras in quantum theory']

    del tree['Partial differential equations']['Equations of mathematical physics and other areas of application']
    del tree['Statistics']['Sufficiency and information']
    del tree['Functional analysis']['Other (nonclassical) types of functional analysis']
    del tree['Functional analysis']['Miscellaneous applications of functional analysis']

    for k_top, top in tree.items():
        for k_2, v in top.items():
            if not v:
                del top[k_2]
            elif k_2 == u'Applications':
                del top[k_2]
            elif 'proceedings' in k_2.lower() or 'conferences' in k_2.lower() or 'collections' in k_2.lower():
                del top[k_2]

    return tree


def read_pacs(path):
    import re

    tree_pacs = {}

    top_top_parent = None
    top_parent = None
    parent = None

    pacs_file = file(path)

    see_also_re = re.compile('\(see also.+?\)')
    for_see_re = re.compile('\(for.+?see.+?\)')
    tags_re = re.compile('<[^ ].*?>')

    for line in pacs_file:
        if line.startswith('#'):
            continue
        if not line.strip():
            continue

        line = line.strip()
        code = line[0:8]
        if code.strip() == '... ...':
            continue

        if code == 'APPENDIX':
            break

        name = line[9:].decode('utf-8').replace('$', '')

        name = see_also_re.sub('', name)
        name = for_see_re.sub('', name)
        name = tags_re.sub('', name)

        codes = code.split('.')
        if len(codes) < 3:
            print code
        is_top_level =  False

        if codes[0].isdigit():
            top_code = int(codes[0])
            is_top_level = (top_code % 10 == 0) & (codes[1] == '00') & (codes[2] == '00')

        if is_top_level:
            top_top_parent = name
            tree_pacs[top_top_parent] = {}
        elif (codes[1] == '00') & (codes[2] == '00'):
            top_parent = name
            tree_pacs[top_top_parent][top_parent] = {}
        elif codes[2][0] in ['+', '-']:
            parent = name
            tree_pacs[top_top_parent][top_parent][parent] = []
        else: # tabs == 0
            tree_pacs[top_top_parent][top_parent][parent].append(name)

    del tree_pacs['GENERAL']

    pacs = {}
    for k_0, cat_top in tree_pacs.items():
        for k_1, cat in cat_top.items():
            pacs[k_0 + ' ' + k_1] = cat

    general_pacs = {}
    for k_0, cat_top in tree_pacs.items():
        general_pacs[k_0] = {}

        for k_1, cat in cat_top.items():
            desc = []
            for k_2, low_cat in cat.items():
                desc.append(k_2)
                desc.extend(low_cat)

            general_pacs[k_0][k_1] = desc

    return general_pacs

def read_acm(path):
    import rdflib
    from rdflib.namespace import SKOS

    logging.getLogger('rdflib.term').setLevel(logging.ERROR)

    g = rdflib.Graph()
    g.load(path)

    pref_len = len('file:///' + path)

    acm_titles = {}

    for s, p, o in g.triples((None, SKOS.prefLabel, None)):
        if o.language != 'en':
            continue
        acm_titles[s[pref_len:]] = o.value

    broader = defaultdict(set)
    narrower = defaultdict(set)

    for s, _, o in g.triples((None, SKOS.broader, None)):
        subj_title = acm_titles[s[pref_len:]]
        obj_title = acm_titles[o[pref_len:]]
        broader[subj_title].add(obj_title)
        narrower[obj_title].add(subj_title)

    top_level = ['Hardware', 'Computer systems organization', 'Networks', 
                 'Software and its engineering', 'Theory of computation', 
                 'Information systems', 'Security and privacy', 
                 'Human-centered computing', 'Computing methodologies']

    acm_tree = {}

    def dfs(result, cat):
        result.append(cat)
        if cat in narrower:
            for n_cat in narrower[cat]:
                dfs(result, n_cat)
        return result

    for most_top in top_level:
        acm_tree[most_top] = {}
        for cat in narrower[most_top]:
            acm_tree[most_top][cat] = dfs([], cat)

    return acm_tree



def read(scheme, path):
    log.debug('reading "%s" scheme from %s' % (scheme, path))

    if scheme == 'pacs':
        return read_pacs(path)
    elif scheme == 'msc':
        return read_msc(path)
    elif scheme == 'acm':
        return read_acm(path)
    raise Exception('unknown classification scheme')


def merge(schemes):
    all_cat_tree = defaultdict(dict)

    for d in schemes:
        for k, dict_inner in d.items():
            all_cat_tree[k].update(dict_inner)

    return all_cat_tree
