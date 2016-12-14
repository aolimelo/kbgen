import warnings
from math import floor

import numpy as np
from scipy.stats import pareto, zipf, powerlaw, uniform, expon, foldnorm, truncexpon, truncnorm

from load_tensor_tools import loadGraphNpz, loadTypesNpz
from kb_models.model_m1 import KBModelM1
from util import MultiType

models_dict = {"pareto":pareto,
               "zipf":zipf,
               "powerlaw":powerlaw,
               "uniform":uniform,
               "expon":expon,
               "foldnorm":foldnorm,
               "truncexpon":truncexpon,
               "truncnorm":truncnorm}

class KBModelEMi(KBModelM1):
    def __init__(self, model, dist_subjects, dist_objects):
        assert isinstance(model, KBModelM1)
        self.base_model = model
        for k,v in model.__dict__.items():
            self.__dict__[k] = v
        self.dist_subjects = dist_subjects
        self.dist_objects = dist_objects

    def select_instance(self, n, model=None):
        if model is not None:
            d, params = model
            model = models_dict[d]
            arg = params[:-2]
            scale = params[-1]
            idx = model.rvs(size=1, loc=0, scale=min([n,scale/self.step]), *arg)[0]
            return int(floor(min([idx,n-1])))
        else:
            return super(KBModelEMi, self).select_instance(n)

    def select_subject_model(self, r, domains):
        if hasattr(self,"inv_functionalities") and self.inv_functionalities[r] == 1:
            return None
        if domains in self.dist_subjects[r]:
            return self.dist_subjects[r][domains]
        else:
            return None


    def select_object_model(self, r, domains, ranges):
        if hasattr(self,"functionalities") and self.functionalities[r] == 1:
            return None
        if ranges in self.dist_objects[r]:
            return self.dist_objects[r][ranges]
        else:
            return None

    @staticmethod
    def learn_best_dist_model(dist, distributions=[truncexpon]):
        dist.sort(reverse=True)

        x = range(len(dist))
        y = np.array(dist)
        y = (y.astype(float) / np.sum(y)).tolist()

        uniform_y = np.full(len(dist), 1.0/len(dist), dtype=float)
        best_sse = np.sum(np.power(y - uniform_y, 2.0))
        best_model = None
        if best_sse > 0:
            for d in distributions:
                warnings.filterwarnings('ignore')
                # fit dist to data
                if sum(dist) > 1000:
                    data_sample = np.random.choice(x, 1000, replace=True, p=y)
                    params = d.fit(data_sample, loc=0, scale=len(dist))
                else:
                    data = []
                    for i, freq in enumerate(dist):
                        data = data + [i] * int(freq)
                    params = d.fit(data,loc=0,scale=len(dist))

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = d.pdf(x, loc=0, scale=len(dist), *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                if sse < best_sse:
                    best_model = (d.__class__.__name__.replace("_gen",""), params)
                    best_sse = sse

        return best_model

    @staticmethod
    def generate_from_tensor(model, input_path, debug=False):
        X = loadGraphNpz(input_path)
        types = loadTypesNpz(input_path).tocsr()
        n_relations = len(X)
        dist_subjects = [{} for r in range(n_relations)]
        dist_objects = [{} for r in range(n_relations)]

        distributions = [truncexpon]

        for r in range(n_relations):
            slice = X[r]
            model.dist_types
            s_sum = {}
            o_sum = {}

            for col in slice.col:
                o_sum[col] = 1 if col not in o_sum else o_sum[col]+1
            for row in slice.row:
                s_sum[row] = 1 if row not in s_sum else s_sum[row]+1

            for s, count in s_sum.items():
                s_types = MultiType(types[s].indices)
                assert s_types in model.dist_types
                if s_types not in dist_subjects[r]:
                    dist_subjects[r][s_types] = []
                dist_subjects[r][s_types].append(count)

            for o,count in o_sum.items():
                o_types = MultiType(types[o].indices)
                assert o_types in model.dist_types
                if o_types not in dist_objects[r]:
                    dist_objects[r][o_types] = []
                dist_objects[r][o_types].append(count)

            models_subjects = [{} for r in range(n_relations)]
            models_objects = [{} for r in range(n_relations)]

            for r in range(n_relations):
                for k,dist in dist_subjects[r].items():
                    models_subjects[r][k] = KBModelEMi.learn_best_dist_model(dist_subjects[r][k], distributions)
                for k,dist in dist_objects[r].items():
                    models_objects[r][k] = KBModelEMi.learn_best_dist_model(dist_objects[r][k], distributions)

        return KBModelEMi(model, models_subjects, models_objects)

    def synthesize(self, size=1.0, ne=None, nf=None, debug=False, pca=True):
        return self.base_model.synthesize(size, ne, nf, debug, pca)