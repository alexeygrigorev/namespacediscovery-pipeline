'''
Created on Nov 8, 2015

@author: alexey
'''

import pickle
import json
import codecs

import luigi
from luigi.file import LocalTarget


from nd.read import mlp_read, scheme
from nd.read import gold_standard

import nd.algorithms.ivs as ivs

from nd.algorithms.cluster import cluster
from nd.algorithms.dimred import dimred

from nd.algorithms import namespace
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
    stemmer = luigi.Parameter()

    def run(self):
        with open(self.input().path, 'rb') as f:
            mlp_data = pickle.load(f)

        mlp_isv = ivs.process(self.type, self.stemmer, mlp_data)

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


class LoadReferenceCategoriesTask(luigi.Task):
    data_dir = luigi.Parameter()
    cached_result = luigi.Parameter()

    def run(self):
        msc = scheme.read('msc', self.data_dir + '/msc.txt')
        pacs = scheme.read('pacs', self.data_dir + '/pacs.txt')
        acm = scheme.read('acm', self.data_dir + '/acm_skos_taxonomy.xml')

        merged_scheme = scheme.merge([msc, pacs, acm])

        with open(self.output().path, 'wb') as f:
            pickle.dump(merged_scheme, f)

    def output(self):
        return LocalTarget(self.cached_result)


class BuildNamespacesTask(luigi.Task):
    cached_result = luigi.Parameter()

    purity_threshold = luigi.FloatParameter(default=0.8)
    min_size = luigi.IntParameter(default=3)
    format = luigi.Parameter('json')

    def run(self):
        with open(self.input()['mlp'].path, 'rb') as f:
            mlp_data = pickle.load(f)
        with open(self.input()['cluster'].path, 'rb') as f:
            labels = pickle.load(f)
        with open(self.input()['ref'].path, 'rb') as f:
            merged_scheme = pickle.load(f)

        evaluator = create_evaluator(mlp_data)
        pure_clusters = evaluator.high_purity_clusters(labels,
                threshold=self.purity_threshold, min_size=self.min_size)

        ROOT = assign_clusters_to_scheme(merged_scheme, labels, mlp_data,
                                         evaluator, pure_clusters)
        dto = ROOT.to_dict(evaluator)

        with codecs.open(self.output().path, 'w', 'utf-8') as f:
            if self.format == 'json':
                json.dump(dto, f, indent=2, sort_keys=True, ensure_ascii=False)
            elif self.format == 'wiki':
                ROOT.to_wiki(evaluator, f)

    def requires(self):
        return {'ref': LoadReferenceCategoriesTask(),
                'mlp': IdentifierVsmRepresentationTask(),
                'cluster': ClusteringTask()}

    def output(self):
        return LocalTarget(self.cached_result)


class NamespacesForGoldStandardTask(luigi.Task):
    cached_result = luigi.Parameter()
    gold_standard_file = luigi.Parameter()

    def run(self):
        with open(self.input()['mlp'].path, 'rb') as f:
            mlp_data = pickle.load(f)
        with open(self.input()['cluster'].path, 'rb') as f:
            labels = pickle.load(f)
        with open(self.input()['ref'].path, 'rb') as f:
            merged_scheme = pickle.load(f)

        evaluator = create_evaluator(mlp_data)

        doc_titles = gold_standard.read_titles(self.gold_standard_file)
        dtos = namespace.namespaces_for_titles(merged_scheme, labels, evaluator,
                                                     mlp_data, doc_titles)
        with codecs.open(self.output().path, 'w', 'utf-8') as f:
            json.dump(dtos, f, indent=2, sort_keys=True, ensure_ascii=False)


    def requires(self):
        return {'ref': LoadReferenceCategoriesTask(),
                'mlp': IdentifierVsmRepresentationTask(),
                'cluster': ClusteringTask()}

    def output(self):
        return LocalTarget(self.cached_result)


if __name__ == "__main__":
    import logging
    logging.basicConfig()
    logging.getLogger('nd.read').setLevel(logging.DEBUG)
    logging.getLogger('nd.algorithms').setLevel(logging.DEBUG)
    logging.getLogger('nd.utils').setLevel(logging.DEBUG)

    luigi.run(main_task_cls=NamespacesForGoldStandardTask, local_scheduler=True)