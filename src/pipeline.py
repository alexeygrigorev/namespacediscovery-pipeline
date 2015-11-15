'''
Created on Nov 8, 2015

@author: alexey
'''

import pickle

import luigi
from luigi.file import LocalTarget

from nd.flow.extra import InMemoryTask

from nd.read import mlp_read, scheme


import nd.algorithms.ivs as ivs

from nd.algorithms.cluster import cluster
from nd.algorithms.dimred import dimred
from nd.algorithms.evaluate import create_evaluator


class MlpResultsReadTask(luigi.Task):
    mlp_results = luigi.Parameter()
    categories_processed = luigi.Parameter()
    cached_result = luigi.Parameter()

    def run(self):
        props = {'input': self.mlp_results,
                 'categories': self.categories_processed }
        mlp_data = mlp_read.main_read(props)

        with open(self.output().path, 'wb') as f:
            pickle.dump(mlp_data, f)

    def output(self):
        return LocalTarget(self.cached_result)


class IdentifierVsmRepresentationTask(luigi.Task):
    cached_result = luigi.Parameter()
    type = luigi.Parameter()

    def run(self):
        with open(self.input().path, 'rb') as f:
            mlp_data = pickle.load(f)

        mlp_isv = ivs.process(self.type, mlp_data)

        with open(self.output().path, 'wb') as f:
            pickle.dump(mlp_isv, f)

    def requires(self):
        return MlpResultsReadTask()

    def output(self):
        return LocalTarget(self.cached_result)


class VectorizerTask(luigi.Task):
    cached_result = luigi.Parameter()

    use_idf = luigi.BooleanParameter(default=True)
    sublinear_tf = luigi.BooleanParameter(default=True)

    dim_red = luigi.Parameter(default='svd')
    dim_red_N = luigi.IntParameter(default=1000)

    def run(self):
        with open(self.input().path, 'rb') as f:
            mlp_isv = pickle.load(f)

        vectors = ivs.vectorize(mlp_isv, use_idf=self.use_idf, sublinear_tf=self.use_idf)
        vectors = dimred(vectors, algo=self.dim_red, N=self.dim_red_N)

        with open(self.output().path, 'wb') as f:
            pickle.dump(vectors, f)

    def requires(self):
        return IdentifierVsmRepresentationTask()

    def output(self):
        return LocalTarget(self.cached_result)


class ClusteringTask(luigi.Task):
    cached_result = luigi.Parameter()

    algorithm = luigi.Parameter()
    params = luigi.Parameter()

    def run(self):
        params = [param_val.split('=') for param_val in self.params.split(',')]
        kwargs = { key: value for key, value in params }

        with open(self.input().path, 'rb') as f:
            vectors = pickle.load(f)

        labels = cluster(vectors, algo=self.algorithm, **kwargs)

        with open(self.output().path, 'wb') as f:
            pickle.dump(labels, f)

    def requires(self):
        return VectorizerTask()

    def output(self):
        return LocalTarget(self.cached_result)

if __name__ == "__main__":
    import logging
    logging.basicConfig()
    logging.getLogger('nd.read').setLevel(logging.DEBUG)
    logging.getLogger('nd.algorithms').setLevel(logging.DEBUG)
    logging.getLogger('nd.utils').setLevel(logging.DEBUG)

    luigi.run(main_task_cls=ClusteringTask, local_scheduler=True)