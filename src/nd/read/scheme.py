'''
Created on Nov 8, 2015

@author: alexey
'''
from collections import defaultdict

import logging
log = logging.getLogger('nd.read')


class ClassificationCategory():
    full_name = None
    code = None
    source = None

    def to_dict(self):
        return {'full_name': self.full_name, 'code': self.code, 'source': self.source}

    def __hash__(self):
        return hash((self.full_name, self.source))

    def __eq__(self, o):
        return (self.full_name, self.source) == (o.full_name, o.source)

    def __repr__(self):
        return '%s (%s/%s)' % (self.full_name, self.code, self.source)

def new_Category(full_name, code, source):
    cat = ClassificationCategory()
    cat.full_name = full_name
    cat.code = code
    cat.source = source
    return cat

# because now keys are the category objects, we need helper methods to retrieve/delete
# the items from the dictionaries by name

def del_item_with_names(d, names):
    for k in d.keys():
        if k.full_name in names:
            del d[k]

def get(d, name):
    for k, v in d.items():
        if k.full_name == name:
            return v
    raise KeyError(name)


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
            top_parent = new_Category(name, code, 'MSC')
            tree[top_parent] = {}
        elif tabs == 1:
            parent = new_Category(name, code, 'MSC')
            tree[top_parent][parent] = []
        else:
            category = new_Category(name, code, 'MSC')
            tree[top_parent][parent].append(category)

    del_item_with_names(tree, ['General', 'History and biography', 'Mathematics education'])
    del_item_with_names(get(tree, 'Quantum theory'), ['Axiomatics, foundations, philosophy', 
        'Applications to specific physical systems', 'Groups and algebras in quantum theory'])

    del_item_with_names(get(tree, 'Partial differential equations'), 
        ['Equations of mathematical physics and other areas of application'])
    del_item_with_names(get(tree, 'Statistics'), 
        ['Sufficiency and information'])
    del_item_with_names(get(tree, 'Functional analysis'), 
        ['Other (nonclassical) types of functional analysis', 
         'Miscellaneous applications of functional analysis'])

    for k_top, top in tree.items():
        for k_2, list_subcats in top.items():
            if not list_subcats:
                del top[k_2]
                continue

            k_2_name = k_2.full_name.lower()
            if k_2_name == u'Applications':
                del top[k_2]
            elif 'proceedings' in k_2_name or 'conferences' in k_2_name or 'collections' in k_2_name:
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
            top_top_parent = new_Category(name, code, 'PACS')
            tree_pacs[top_top_parent] = {}
        elif (codes[1] == '00') & (codes[2] == '00'):
            top_parent = new_Category(name, code, 'PACS')
            tree_pacs[top_top_parent][top_parent] = {}
        elif codes[2][0] in ['+', '-']:
            parent = new_Category(name, code, 'PACS')
            tree_pacs[top_top_parent][top_parent][parent] = []
        else: # tabs == 0
            category = new_Category(name, code, 'PACS')
            tree_pacs[top_top_parent][top_parent][parent].append(category)

    del_item_with_names(tree_pacs, 'GENERAL')

    # let's combine 3rd and 4th levels 
    result = {}
    for k_0, cat_top in tree_pacs.items():
        result[k_0] = {}

        for k_1, cat in cat_top.items():
            desc = []
            for k_2, low_cat in cat.items():
                desc.append(k_2)
                desc.extend(low_cat)

            result[k_0][k_1] = desc

    return result

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

    def dfs(result, cat):
        result.append(cat)
        if cat in narrower:
            for n_cat in narrower[cat]:
                dfs(result, n_cat)
        return result

    def acm_cat(name):
        return new_Category(name, 'NO_CODE', 'ACM')

    acm_tree = {}

    for most_top in top_level:
        mt_cat = acm_cat(most_top)
        acm_tree[mt_cat] = {}
        for cat in narrower[most_top]:
            sub_cats = dfs([], cat)
            acm_tree[mt_cat][acm_cat(cat)] = [acm_cat(c) for c in sub_cats]

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
