import logging
import random
from copy import deepcopy

import numpy as np
import tqdm
from matplotlib import pyplot as plt
from kb_models.model_m2 import KBModelM2
from numpy.random import choice
from rdflib import Graph

from util import URIEntity, URIRelation, normalize, create_logger


class KBModelM3(KBModelM2):
    '''
    Model based on the KBModelM2 and containing horn rules for synthesizing facts
    - Since the original distribution can be affected by the facts produced by rules, we keep
     the counts of the distributions and decrease them for every added fact, in order to make
      sure the original distribution is not disturbed.
    - Horn rules are assumed to be produced by AMIE
    '''
    def __init__(self, model_naive, rules):
        assert type(model_naive) == KBModelM2
        for k,v in model_naive.__dict__.items():
            self.__dict__[k] = v
        self.rules = rules
        self.max_recurr = 10
        self.pca = True
        pass

    def print_synthesis_details(self):
        ''' Prints synthesis statistics for debugging purposes '''
        super(KBModelM3, self).print_synthesis_details()
        self.logger.debug("exhausted: %d" % self.count_exhausted_facts)
        self.logger.debug("no entities of type: %d" % self.count_no_entities_of_type)
        self.logger.debug("key error %d" % self.key_error_count)
        self.logger.debug("added by rules: %d" % self.count_rule_facts)
        self.logger.debug("step %f" % self.step)
        self.logger.debug(self.d_r)

    def plot_histogram(self,vals,weights):
        '''
        Plots a histrogram of precalculated counts (display distribution)
        :param vals: values of the bins (x axis)
        :param weights: counts of each bin (y axis)
        '''
        x = np.array(vals)
        y = np.array(weights)
        y /= np.sum(y)

        plt.figure(0)
        n, bins, patches = plt.hist(x,weights=y,bins=len(vals), normed=1, facecolor='green', alpha=0.75)
        plt.show()

    def start_counts(self):
        '''Initializes various counts'''
        self.count_facts = 0
        self.count_rule_facts = 0
        self.count_exhausted_facts = 0
        self.count_already_existent_facts = 0
        self.count_no_entities_of_type = 0
        self.count_violate_functionality_facts = 0
        self.count_violate_inv_functionality_facts = 0
        self.count_violate_non_reflexiveness_facts = 0
        self.key_error_count = 0

    def validate_owl(self,r_i,s_id,o_id):
        '''
        Validates functionality, inverse functionality and non-reflexiveness for a given fact
        :param r_i: relation id
        :param s_id: subject id
        :param o_id: object id
        :return: true if consistent, false otherwise
        '''
        if r_i in self.func_rel_subj_pool:
            if s_id not in self.func_rel_subj_pool[r_i]:
                self.count_violate_functionality_facts += 1
                return False

        if r_i in self.inv_func_rel_subj_pool:
            if o_id not in self.inv_func_rel_subj_pool[r_i]:
                self.count_violate_inv_functionality_facts += 1
                return False

        if not self.reflexiveness[r_i]:
            if s_id == o_id:
                self.count_violate_non_reflexiveness_facts += 1
                return False

        return True

    def produce_rules(self, g, r_i, fact, recurr_count=0):
        '''
        Checks if a new fact added trigger any horn rules to generate new facts
        :param g: synthesized graph
        :param r_i: id of the added relation
        :param fact: added fact
        :param recurr_count: recursion count (tracks the recursion depth)
        '''
        s,p,o = fact
        if recurr_count<self.max_recurr and r_i in self.rules.rules_per_relation:
            rules = self.rules.rules_per_relation[r_i]
            for rule in rules:
                rand_number = random.random()
                if (self.pca and rand_number < rule.pca_conf) or (not self.pca and rand_number < rule.std_conf):
                    new_facts = rule.produce(g,s,p,o)
                    for new_fact in new_facts:
                        s,p,o = new_fact
                        s_id = URIEntity.extract_id(s).id
                        o_id = URIEntity.extract_id(o).id
                        r_i = URIRelation.extract_id(p).id

                        s_types = self.types_entities[s_id]
                        o_types = self.types_entities[o_id]

                        if self.validate_owl(r_i,s_id,o_id) and self.d_rdr[r_i][s_types][o_types] > 0:
                            if self.add_fact(g,new_fact):
                                self.update_distributions(r_i,s_types,o_types)
                                self.update_pools(r_i, s_id, o_id)
                                self.count_rule_facts += 1
                                self.produce_rules(g,r_i,new_fact,recurr_count+1)
                        else:
                            self.count_exhausted_facts += 1

    def update_distributions(self, r_i, s_types, o_types):
        '''
        Updates the distributions after a given fact has been added
        (decreases the step from the original distribution counts)
        :param r_i: id of the relation added
        :param s_types: multitype of the subject
        :param o_types: multitype of the object
        '''
        step = float(self.step)

        if r_i in self.d_r:
            self.d_r[r_i] -= step
            if self.d_r[r_i] <= 0:
                self.delete_relation_entries(r_i)
            if r_i in self.d_dr and s_types in self.d_dr[r_i]:
                self.d_dr[r_i][s_types] -= step
                if self.d_dr[r_i][s_types] <= 0:
                    self.delete_relation_domain_entries(r_i,s_types)
                if s_types in self.d_rdr[r_i] and o_types in self.d_rdr[r_i][s_types]:
                    self.d_rdr[r_i][s_types][o_types] -= step
                    if self.d_rdr[r_i][s_types][o_types] <= 0:
                        self.delete_relation_domain_range_entries(r_i,s_types,o_types)

        self.delete_empty_entries(r_i,s_types)


    def prune_distrbutions(self):
        '''
        Prunes the original distributions based on the entities generated.
        If a given multitype does not have any instances in the synthesized data, all the entries
        in the distributions containing the given multitype are deleted
        '''
        counts_removed = 0.0
        for r in self.d_r.keys():
            if self.d_r[r] == 0:
                self.delete_relation_entries(r)
            for domain in self.d_dr[r].keys():
                if domain not in self.entities_types.keys() or not self.entities_types[domain]:
                    counts_removed += self.d_dr[r][domain]
                    self.delete_relation_domain_entries(r,domain)
                else:
                    for range in set(self.d_rdr[r][domain].keys()):
                        if range not in self.entities_types.keys() or not self.entities_types[range]:
                            counts_removed += self.d_rdr[r][domain][range]
                            self.delete_relation_domain_range_entries(r,domain,range)

                    if not self.d_rdr[r][domain]:
                        counts_removed += self.d_dr[r][domain]
                        self.delete_relation_domain_entries(r,domain)

            if not self.d_dr[r]:
                counts_removed += self.d_r[r]
                self.delete_relation_entries(r)

        if counts_removed > 0:
            self.step *= (self.n_facts-counts_removed)/self.n_facts

    def delete_relation_entries(self,r_i):
        if r_i in self.d_r:
            del self.d_r[r_i]
        if r_i in self.d_dr:
            del self.d_dr[r_i]
        if r_i in self.d_rdr:
            del self.d_rdr[r_i]

    def delete_relation_domain_entries(self,r_i,domain):
        if r_i in self.d_rdr and domain in self.d_rdr[r_i]:
            del self.d_rdr[r_i][domain]
        if r_i in self.d_dr and domain in self.d_dr[r_i]:
            del self.d_dr[r_i][domain]

    def delete_relation_domain_range_entries(self,r_i,domain,range):
        if r_i in self.d_rdr and domain in self.d_rdr[r_i] and range in self.d_rdr[r_i][domain]:
            del self.d_rdr[r_i][domain][range]

    def delete_empty_entries(self,r_i,s_types):
        counts_removed = 0
        if r_i in self.d_dr and not self.d_dr[r_i]:
            counts_removed += self.d_r[r_i]
            self.delete_relation_entries(r_i)

        if r_i in self.d_rdr and s_types in self.d_rdr[r_i] and not self.d_rdr[r_i][s_types]:
            counts_removed += self.d_dr[r_i][s_types]
            self.delete_relation_domain_entries(r_i, s_types)

        if counts_removed > 0:
            remaining_facts = float(self.n_entities - self.count_facts) * self.step
            self.step *= (remaining_facts - counts_removed) / remaining_facts

    def update_pools(self, r_i, s_i, o_i):
        counts_removed = 0
        if r_i in self.d_r:
            if r_i in self.func_rel_subj_pool and s_i in self.func_rel_subj_pool[r_i]:
                self.func_rel_subj_pool[r_i].remove(s_i)
            if r_i in self.inv_func_rel_subj_pool and o_i in self.inv_func_rel_subj_pool[r_i]:
                self.inv_func_rel_subj_pool[r_i].remove(o_i)

            if (r_i in self.func_rel_subj_pool and not self.func_rel_subj_pool[r_i]) or \
               (r_i in self.inv_func_rel_subj_pool and not self.inv_func_rel_subj_pool[r_i]):
                counts_removed += self.d_r[r_i] if r_i in self.d_r else 0
                self.delete_relation_entries(r_i)

        if counts_removed>0:
            remaining_facts = float(self.synthetic_facts - self.count_facts) * self.step
            self.step *= (remaining_facts - counts_removed) / remaining_facts

    def adjust_func_inv_func_relations(self):
        counts_removed = 0

        for r_i in self.functionalities.keys():
            if r_i in self.d_r and r_i in self.func_rel_subj_pool:
                diff = self.d_r[r_i] - len(self.func_rel_subj_pool[r_i])
                if diff > 0:
                    self.d_r[r_i] = len(self.func_rel_subj_pool[r_i])
                    counts_removed += diff

        for r_i in self.inv_func_rel_subj_pool.keys():
            if r_i in self.d_r and r_i in self.inv_func_rel_subj_pool:
                diff = self.d_r[r_i] - len(self.inv_func_rel_subj_pool[r_i])
                if diff > 0:
                    self.d_r[r_i] = len(self.inv_func_rel_subj_pool[r_i])
                    counts_removed += diff

            if counts_removed>0:
                remaining_facts = float(self.synthetic_facts - self.count_facts) * self.step
                self.step *= (remaining_facts - counts_removed) / remaining_facts

        return self.d_r

    def synthesize(self, size=1.0, ne=None, nf=None, debug=False, pca=True):
        print("Synthesizing HORN model")

        if debug:
            self.logger = create_logger(logging.DEBUG)
        else:
            self.logger = create_logger(logging.INFO)

        self.pca = pca

        self.start_counts()

        g = Graph()

        # self.step = 1
        self.step = 1.0/float(size)

        self.logger.info("%d \tentities and %d \tfacts \t on original dataset" % (self.n_entities,self.n_facts))
        self.synthetic_facts = self.n_facts
        self.synthetic_entities = self.n_entities

        self.synthetic_entities = int(self.n_entities/self.step)
        self.synthetic_facts = int(self.n_facts/self.step)
        if ne is not None:
            self.synthetic_entities = ne
        if nf is not None:
            self.synthetic_facts = nf

        self.step /= 1.1

        self.logger.info("%d \tentities and %d \tfacts \t on synthetic dataset" % (self.synthetic_entities, self.synthetic_facts))
        quadratic_relations = self.check_for_quadratic_relations()
        adjusted_dist_relations = self.adjust_quadratic_relation_distributions(deepcopy(self.dist_relations), quadratic_relations)

        self.inv_rel_dict = {k:v for v,k in self.rel_dict.items()}

        self.logger.debug("quadratic relations: %s"%[self.inv_rel_dict[r] for r in quadratic_relations])
        g = self.synthesize_types(g,self.n_types)
        g = self.synthesize_relations(g,self.n_relations)
        g = self.synthesize_schema(g)
        g, entities_types = self.synthesize_entities(g,self.synthetic_entities)

        self.types_entities = {k:v for v in entities_types.keys() for k in entities_types[v]}
        self.entities_types = entities_types

        self.logger.debug("copying distributions")
        self.d_r = deepcopy(adjusted_dist_relations)
        self.d_dr = deepcopy(self.dist_domains_relation)
        self.d_rdr = deepcopy(self.dist_ranges_domain_relation)

        self.logger.debug("pruning distributions for domain and ranges of types without instances")
        self.prune_distrbutions()

        self.func_rel_subj_pool = self.functional_rels_subj_pool()
        self.inv_func_rel_subj_pool = self.invfunctional_rels_subj_pool()
        self.d_r = self.adjust_func_inv_func_relations()

        self.saturated_subj = {r_i:{} for r_i in range(self.n_relations)}
        self.saturated_obj = {r_i:{} for r_i in range(self.n_relations)}

        self.logger.debug("func rels subj pool = %s"%{r:len(ents) for r,ents in self.func_rel_subj_pool.items()})
        self.logger.debug("inv func rels subj pool = %s" %{r:len(ents) for r,ents in self.inv_func_rel_subj_pool.items()})

        self.logger.info("synthesizing facts")
        self.pbar = tqdm.tqdm(total=self.synthetic_facts)
        while self.count_facts < self.synthetic_facts and self.d_r:
            r_i = choice(self.d_r.keys(), 1, normalize(self.d_r.values()))[0]
            s_types = None
            o_types = None
            s_i = o_i = -1
            n_entities_subject = n_entities_object = -1
            self.logger.debug("relation %d = %s"%(r_i,self.inv_rel_dict[r_i]))

            if r_i in self.d_dr and self.d_dr[r_i]:
                s_types = choice(self.d_dr[r_i].keys(), 1, normalize(self.d_dr[r_i].values()))[0]
                subject_entities = set(entities_types[s_types])
                if r_i in self.func_rel_subj_pool:
                    subject_entities = subject_entities.intersection(self.func_rel_subj_pool[r_i])

                n_entities_subject = len(subject_entities)

                if n_entities_subject > 0 and s_types in self.d_rdr[r_i] and self.d_rdr[r_i][s_types]:
                    o_types = choice(self.d_rdr[r_i][s_types].keys(), 1, normalize(self.d_rdr[r_i][s_types].values()))[0]
                    object_entities = set(entities_types[o_types])
                    if r_i in self.inv_func_rel_subj_pool:
                        object_entities = object_entities.intersection(self.inv_func_rel_subj_pool[r_i])

                    if s_types in self.saturated_obj[r_i]:
                        object_entities = object_entities - self.saturated_obj[r_i][s_types]

                    if o_types in self.saturated_subj[r_i]:
                        subject_entities = subject_entities - self.saturated_subj[r_i][o_types]

                    # ensures non-reflexiveness by removing subject id from objects pool
                    if not self.reflexiveness[r_i] and s_i in object_entities:
                        object_entities.remove(s_i)

                    n_entities_subject = len(subject_entities)
                    n_entities_object = len(object_entities)

                    if n_entities_object > 0 and n_entities_subject > 0:

                        subject_model = self.select_subject_model(r_i, s_types)
                        s_pool_i = self.select_instance(n_entities_subject, subject_model)
                        s_i = list(subject_entities)[s_pool_i]

                        object_model = self.select_object_model(r_i, s_types, o_types)
                        o_pool_i = self.select_instance(n_entities_object, object_model)
                        o_i = list(object_entities)[o_pool_i]

                        s = URIEntity(s_i).uri
                        o = URIEntity(o_i).uri
                        p = URIRelation(r_i).uri
                        fact = (s,p,o)
                        try:
                            o_offset,s_offset = 0,0
                            if random.random() < 0.5:
                                while not self.add_fact(g, (s, p, o)) and o_offset < len(object_entities):
                                    # try to add triple with other objects
                                    o_offset += 1
                                    o_i = list(object_entities)[(o_pool_i + o_offset) % len(object_entities)]
                                    o = URIEntity(o_i).uri

                                if o_offset >= len(object_entities):  # fact could not be added
                                    # impossible to add facts for r_i with subject s_i
                                    if o_types not in self.saturated_subj[r_i]:
                                        self.saturated_subj[r_i][o_types] = set()
                                    self.saturated_subj[r_i][o_types].add(s_i)
                            else:
                                o_i = list(object_entities)[o_pool_i]
                                o = URIEntity(o_i).uri
                                while not self.add_fact(g, (s, p, o)) and s_offset < len(subject_entities):
                                    # try to add fact with other subjects
                                    s_offset += 1
                                    s_i = list(subject_entities)[(s_pool_i + s_offset) % len(subject_entities)]
                                    s = URIEntity(s_i).uri
                                if s_offset >= len(subject_entities):
                                    #impossible to add facts for r_i with object o_i
                                    if s_types not in self.saturated_obj[r_i]:
                                        self.saturated_obj[r_i][s_types] = set()
                                    self.saturated_obj[r_i][s_types].add(o_i)

                            if o_offset < len(object_entities) and s_offset < len(subject_entities):
                                self.update_distributions(r_i,s_types,o_types)
                                self.produce_rules(g,r_i,(s,p,o))
                                self.update_pools(r_i, s_i, o_i)
                                continue
                        except KeyError:
                            self.key_error_count += 1
                    else:
                        self.count_no_entities_of_type += 1
                else:
                    self.count_exhausted_facts += 1
            else:
                self.count_exhausted_facts += 1

            if n_entities_object == 0:
                self.delete_relation_domain_entries(r_i, s_types)
            if n_entities_object == 0:
                self.delete_relation_domain_range_entries(r_i, s_types, o_types)
            self.delete_empty_entries(r_i,s_types)
            self.update_pools(r_i, s_i, o_i)
            self.print_synthesis_details()

        self.logger.info("synthesized facts = %d from %d"%(self.count_facts,self.synthetic_facts))
        return g
