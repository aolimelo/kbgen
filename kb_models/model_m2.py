from load_tensor_tools import loadGraphNpz
from kb_models.model_m1 import KBModelM1
from rdflib import Graph
from numpy.random import choice
from util import URIEntity, URIRelation, normalize, create_logger
import logging
from scipy.sparse import csr_matrix
import tqdm
import datetime


class KBModelM2(KBModelM1):
    '''
    Model based on the KBModelM1 and containing nonreflexiveness, functionality and inverse funcitonality of relations
    - To avoid violations of the three relation characteristics we keep pools of subjects and entities available for
     each relation. Whenever a fact is generated the subject and the object are removed from their repective pools.
    '''

    def __init__(self, naive_model, functionalities, inv_functionalities, rel_densities, \
                 rel_distinct_subjs, rel_distinct_objs, reflexiveness):
        assert type(naive_model) == KBModelM1
        for k, v in naive_model.__dict__.items():
            self.__dict__[k] = v
        self.functionalities = functionalities
        self.inv_functionalities = inv_functionalities
        self.rel_densities = rel_densities
        self.rel_distinct_subjs = rel_distinct_subjs
        self.rel_distinct_objs = rel_distinct_objs
        self.reflexiveness = reflexiveness

    def print_synthesis_details(self):
        super(KBModelM2, self).print_synthesis_details()
        self.logger.debug("violate func: %d" % self.count_violate_functionality_facts)
        self.logger.debug("violate invfunc: %d" % self.count_violate_inv_functionality_facts)
        self.logger.debug("violate nonreflex: %d" % self.count_violate_non_reflexiveness_facts)

    def valid_functionality(self, g, fact):
        try:
            g.triples((fact[0], fact[1], None)).next()
            self.count_violate_functionality_facts += 1
            return False
        except StopIteration:
            return True

    def valid_inv_functionality(self, g, fact):
        try:
            g.triples((None, fact[1], fact[2])).next()
            self.count_violate_inv_functionality_facts += 1
            return False
        except StopIteration:
            return True

    def valid_reflexiveness(self, g, fact):
        if fact[0] != fact[2]:
            return True
        else:
            self.count_violate_non_reflexiveness_facts += 1
            return False

    def functional_rels_subj_pool(self):
        func_rels_subj_pool = {}
        for r, func in self.functionalities.items():
            if func == 1 and r in self.d_dr:
                func_rels_subj_pool[r] = set()
                for domain in self.d_dr[r]:
                    if domain in self.entities_types.keys():
                        func_rels_subj_pool[r] = func_rels_subj_pool[r].union(set(self.entities_types[domain]))
        return func_rels_subj_pool

    def invfunctional_rels_subj_pool(self):
        invfunc_rels_subj_pool = {}
        for r, inv_func in self.inv_functionalities.items():
            if inv_func == 1 and r in self.d_rdr:
                invfunc_rels_subj_pool[r] = set()
                for domain in self.d_dr[r]:
                    for range in self.d_rdr[r][domain]:
                        if range in self.entities_types.keys():
                            invfunc_rels_subj_pool[r] = invfunc_rels_subj_pool[r].union(set(self.entities_types[range]))
        return invfunc_rels_subj_pool

    def synthesize(self, size=1, ne=None, nf=None, debug=False, pca=True):
        print("Synthesizing OWL model")

        level = logging.DEBUG if debug else logging.INFO
        self.logger = create_logger(level, name="kbgen")
        self.synth_time = create_logger(level, name="synth_time")

        self.step = 1.0 / float(size)
        synthetic_entities = int(self.n_entities / self.step)
        synthetic_facts = int(self.n_facts / self.step)
        if ne is not None:
            synthetic_entities = ne
        if nf is not None:
            synthetic_facts = nf

        g = Graph()

        quadratic_relations = self.check_for_quadratic_relations()
        adjusted_dist_relations = self.adjust_quadratic_relation_distributions(self.dist_relations, quadratic_relations)

        types = range(self.n_types)
        relations = range(self.n_relations)

        g = self.synthesize_types(g, self.n_types)
        g = self.synthesize_relations(g, self.n_relations)
        g = self.synthesize_schema(g)
        g, entities_types = self.synthesize_entities(g, synthetic_entities)
        self.types_entities = {k: v for v in entities_types.keys() for k in entities_types[v]}
        self.entities_types = entities_types

        self.logger.info("synthesizing facts")
        dist_relations = normalize(adjusted_dist_relations.values())

        dist_domains_relation = {}
        for rel in relations:
            dist_domains_relation[rel] = normalize(self.dist_domains_relation[rel].values())

        dist_ranges_domain_relation = {}
        for rel in relations:
            dist_ranges_domain_relation[rel] = {}
            for domain_i in self.dist_ranges_domain_relation[rel].keys():
                dist_ranges_domain_relation[rel][domain_i] = normalize(
                    self.dist_ranges_domain_relation[rel][domain_i].values())

        self.count_facts = 0
        self.count_already_existent_facts = 0
        self.count_violate_functionality_facts = 0
        self.count_violate_inv_functionality_facts = 0
        self.count_violate_non_reflexiveness_facts = 0

        self.logger.info(str(synthetic_facts) + " facts to be synthesized")
        self.pbar = tqdm.tqdm(total=synthetic_facts)
        self.start_t = datetime.datetime.now()
        while self.count_facts < synthetic_facts:
            rel_i = choice(self.dist_relations.keys(), 1, True, dist_relations)[0]
            if rel_i in self.dist_relations.keys():
                # rel_i = self.dist_relations.keys().index(rel_uri)
                # rel_i = i
                domain_i = choice(self.dist_domains_relation[rel_i].keys(), 1, p=dist_domains_relation[rel_i])
                domain_i = domain_i[0]
                n_entities_domain = len(entities_types[domain_i])

                range_i = choice(self.dist_ranges_domain_relation[rel_i][domain_i].keys(),
                                 1, p=dist_ranges_domain_relation[rel_i][domain_i])
                range_i = range_i[0]
                n_entities_range = len(entities_types[range_i])

                if n_entities_domain > 0 and n_entities_range > 0:
                    subject_model = self.select_subject_model(rel_i, domain_i)
                    object_model = self.select_object_model(rel_i, domain_i, range_i)

                    object_i = entities_types[range_i][self.select_instance(n_entities_range, object_model)]
                    subject_i = entities_types[domain_i][self.select_instance(n_entities_domain, subject_model)]

                    p_i = URIRelation(rel_i).uri
                    s_i = URIEntity(subject_i).uri
                    o_i = URIEntity(object_i).uri

                    fact = (s_i, p_i, o_i)
                    if (self.functionalities[rel_i] > 1 or self.valid_functionality(g, fact)) and \
                            (self.inv_functionalities[rel_i] > 1 or self.valid_inv_functionality(g, fact)) and \
                            (self.reflexiveness[rel_i] or self.valid_reflexiveness(g, fact)):
                        self.add_fact(g, fact)

        self.print_synthesis_details()
        self.logger.info("synthesized facts = %d from %d" % (self.count_facts, synthetic_facts))
        return g

    @staticmethod
    def generate_entities_stats(g):
        pass

    @staticmethod
    def generate_from_tensor(naive_model, input_path, debug=False):
        X = loadGraphNpz(input_path)

        functionalities = {}
        inv_functionalities = {}
        reflexiveness = {}
        rel_densities = {}
        rel_distinct_subjs = {}
        rel_distinct_objs = {}
        for p in range(len(X)):
            obj_per_subj = csr_matrix(X[p].sum(axis=1))
            subj_per_obj = csr_matrix(X[p].sum(axis=0))
            functionalities[p] = float(obj_per_subj.sum()) / obj_per_subj.nnz
            inv_functionalities[p] = float(subj_per_obj.sum()) / subj_per_obj.nnz
            reflexiveness[p] = X[p].diagonal().any()
            rel_distinct_subjs[p] = obj_per_subj.nnz
            rel_distinct_objs[p] = subj_per_obj.nnz
            rel_densities[p] = float(X[p].nnz) / (rel_distinct_subjs[p] * rel_distinct_objs[p])

        owl_model = KBModelM2(naive_model, functionalities, inv_functionalities, rel_densities, \
                              rel_distinct_subjs, rel_distinct_objs, reflexiveness)

        return owl_model
