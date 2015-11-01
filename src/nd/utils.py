'''
Created on Nov 1, 2015

@author: alexey
'''
from time import time


def time_it(function):
    def wrapper(*args, **kwargs):
        t0 = time()
        function(*args, **kwargs)
        taken_time = time() - t0
        print "done in %0.5fs." % (taken_time)
    return wrapper