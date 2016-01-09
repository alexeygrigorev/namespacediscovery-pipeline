# -*- coding: utf-8 -*-


import json

import logging

log = logging.getLogger('nd.read')


def read_titles(file_path):
    """
    :param file_path: path to the gold standard json
    :return: list of titles extracted from the gold standard json
    """
    with file(file_path, 'r') as json_file:
        def_list = json.load(json_file)
        titles = [d["formula"]["title"] for d in def_list]
        return [t.replace('_', ' ') for t in titles]
