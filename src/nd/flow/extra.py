'''
Created on Nov 15, 2015

@author: alexey
'''

import luigi


class NotDoneTarget(luigi.Target):
    def exists(self):
        return False

class DoneTarget(luigi.Target):
    result = None

    def __init__(self, result):
        self.result = result

    def exists(self):
        return True

class InMemoryTask(luigi.Task):
    target = NotDoneTarget()

    def run(self):
        result = self.calculate()
        self.target = DoneTarget(result)

    def output(self):
        return self.target