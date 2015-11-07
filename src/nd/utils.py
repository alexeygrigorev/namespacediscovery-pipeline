'''
Created on Nov 1, 2015

@author: alexey
'''
from time import time
from functools import wraps

import logging
log = logging.getLogger('nd.utils')


def time_it(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        log.debug("executing %s(%s, %s)" % (function.__name__, args, kwargs))
        t0 = time()
        result = function(*args, **kwargs)
        taken_time = time() - t0
        log.debug("done in %0.5fs." % (taken_time))
        return result

    return wrapper