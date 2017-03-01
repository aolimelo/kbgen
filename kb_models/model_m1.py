from load_tensor_tools import loadGraphNpz, loadTypesNpz, load_types_dict, load_relations_dict, load_type_hierarchy, \
    load_prop_hierarchy, load_domains, load_ranges
from kb_models.model import KBModel
from rdflib import Graph, RDF, OWL, RDFS
from numpy.random import choice, randint
from util import URIEntity, URIRelation, URIType, MultiType, normalize, create_logger
import logging
from scipy.sparse import csr_matrix
import tqdm
import time, datetime


class KBModelM1(KBModel):
    '''
    Simplest knowledge base model composed of the distribution of entities over types and the joint distribution of
    relations, subject and object type sets (represented with the chain rule)
    '''

    def __init__(self, type_hierarchy, prop_hierarchy, domains, ranges, n_entities, n_relations, n_facts, n_types,
                 dist_types, dist_relations,
                 dist_domains_relation, dist_ranges_domain_relation, rel_dict, types_dict=None):
        super(KBModelM1, self).__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_facts = n_facts
        self.n_types = n_types
        self.dist_types = dist_types  # distribution of entities over types
        self.dist_relations = dist_relations  # distribution of facts over relations
        self.dist_domains_relation = dist_domains_relation  # distribution of subject types given relation
        self.dist_ranges_domain_relation = dist_ranges_domain_relation  # distribtution of object types given relation and subject type
        self.rel_dict = rel_dict
        self.types_dict = types_dict
        self.domains = domains
        self.ranges = ranges
        self.type_hierarchy = type_hierarchy
        self.prop_hierarchy = prop_hierarchy
        self.fix_hierarchies()


    def fix_hierarchies(self):
        for i, n in self.type_hierarchy.items():
            n.children = [c if isinstance(c, int) else c.node_id for c in n.children]
            n.parents = [p if isinstance(p, int) else p.node_id for p in n.parents]
        for i, n in self.prop_hierarchy.items():
            n.children = [c if isinstance(c, int) else c.node_id for c in n.children]
            n.parents = [p if isinstance(p, int) else p.node_id for p in n.parents]

    def print_synthesis_details(self):
        self.logger.debug(str(self.count_facts) + " facts added")
        self.logger.debug("already existent: %d" % self.count_already_existent_facts)

    def check_for_quadratic_relations(self):
        quadratic_relations = []
        for r in self.rel_densities.keys():
            func = self.functionalities[r]
            inv_func = self.inv_functionalities[r]
            if func > 10 and inv_func > 10:
                density = self.rel_densities[r]
                n_obj = self.rel_distinct_objs[r]
                n_subj = self.rel_distinct_subjs[r]
                reflex = self.reflexiveness[r]

                if density > 0.1:
                    self.logger.debug(
                        "relation %d:  func=%f,  inv_func=%f,  density=%f,  n_obj=%d,  n_subj=%d,  reflex=%d" % (
                        r, func, inv_func, density, n_obj, n_subj, reflex))
                    quadratic_relations.append(r)
        return quadratic_relations

    def adjust_quadratic_relation_distributions(self, dist_rel, quadratic_relations):
        self.logger.debug("adjusting distribution because of quadratic relations %s"%str(quadratic_relations))
        for r in quadratic_relations:
            dist_rel[r] = dist_rel[r] / self.step
        return dist_rel

    def add_fact(self, g, fact):
        prev_size = len(g)
        g.add(fact)
        if len(g) > prev_size:
            self.pbar.update(1)
            self.count_facts += 1
            if self.count_facts % 10000 == 0:
                self.print_synthesis_details()
            delta = datetime.datetime.now() - self.start_t
            self.synth_time.debug("%d,%d" % (len(g), delta.microseconds / 1000))
            return True
        else:
            self.count_already_existent_facts += 1
            return False

    def synthesize_types(self, g, n_types):
        self.logger.info("synthesizing types")
        for i in range(n_types):
            type_i = URIType(i).uri
            g.add((type_i, RDF.type, OWL.Class))
        self.logger.info(str(n_types) + " types added")
        return g

    def synthesize_schema(self, g):
        self.logger.info("synthesizing schema")
        self.fix_hierarchies()
        if self.type_hierarchy:
            for id, l in self.type_hierarchy.items():
                type_child = URIType(id).uri
                for p_i in l.parents:
                    if not isinstance(p_i, int):
                        p = self.type_hierarchy[p_i]
                        p_i = p.node_id
                    type_parent = URIType(p_i).uri
                    g.add((type_child, RDFS.subClassOf, type_parent))

        if self.prop_hierarchy:
            for id, l in self.prop_hierarchy.items():
                type_child = URIRelation(id).uri
                for p_i in l.parents:
                    if not isinstance(p_i, int):
                        p = self.prop_hierarchy[p_i]
                        p_i = p.node_id
                    type_parent = URIRelation(p_i).uri
                    g.add((type_child, RDFS.subPropertyOf, type_parent))

        if self.domains:
            for r, t in self.domains.items():
                rel = URIRelation(r).uri
                domain = URIType(t).uri
                g.add((rel, RDFS.domain, domain))

        if self.ranges:
            for r, t in self.ranges.items():
                rel = URIRelation(r).uri
                range = URIType(t).uri
                g.add((rel, RDFS.range, range))

        return g

    def synthesize_relations(self, g, n_relations):
        self.logger.info("synthesizing relations")

        for i in range(n_relations):
            rel_i = URIRelation(i).uri
            g.add((rel_i, RDF.type, OWL.ObjectProperty))
        self.logger.info(str(self.n_relations) + " relations added")
        return g

    def synthesize_entities(self, g, n_entities):
        self.logger.info("synthesizing entities")

        entity_types_dict = {}
        for i in self.dist_types.keys():
            entity_types_dict[i] = []
        entity_types = choice(self.dist_types.keys(), n_entities, True, normalize(self.dist_types.values()))
        type_assertions = 0
        pbar = tqdm.tqdm(total=n_entities)
        for i in range(n_entities):
            entity_types_dict[entity_types[i]].append(i)
            entity_i = URIEntity(i).uri
            for t_i in entity_types[i].types:
                type_i = URIType(t_i).uri
                g.add((entity_i, RDF.type, type_i))
                type_assertions += 1
            pbar.update(1)
        self.logger.debug(str(n_entities) + " entities with " + str(type_assertions) + " type assertions added")
        return g, entity_types_dict

    def select_instance(self, n, model=None):
        return randint(n)

    def select_subject_model(self, r, domain):
        return None

    def select_object_model(self, r, domain, range):
        return None

    def synthesize(self, size=1.0, ne=None, nf=None, debug=False, pca=True, ):
        print("Synthesizing NAIVE model")

        level = logging.DEBUG if debug else logging.INFO
        self.logger = create_logger(level, name="kbgen")
        self.synth_time = create_logger(level, name="synth_time")

        self.step = 1.0 / float(size)
        sythetic_entities = int(self.n_entities / self.step)
        synthetic_facts = int(self.n_facts / self.step)

        if ne is not None:
            synthetic_entities = ne
        if nf is not None:
            synthetic_facts = nf

        g = Graph()

        quadratic_relations = self.check_for_quadratic_relations()
        adjusted_dist_relations = self.adjust_quadratic_relation_distributions(self.dist_relations,quadratic_relations)

        types = range(self.n_types)
        relations = range(self.n_relations)

        g = self.synthesize_types(g, self.n_types)
        g = self.synthesize_relations(g, self.n_relations)
        g = self.synthesize_schema(g)
        g, entities_types = self.synthesize_entities(g, sythetic_entities)
        self.types_entities = {k: v for v in entities_types.keys() for k in entities_types[v]}
        self.entities_types = entities_types

        self.logger.info("synthesizing facts")
        dist_relations = normalize(adjusted_dist_relations)

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

                    self.add_fact(g, fact)
        return g

    @staticmethod
    def generate_from_tensor(input_path, debug=False):
        if debug:
            logger = create_logger(logging.DEBUG)
        else:
            logger = create_logger(logging.INFO)

        logger.info("loading data")
        X = loadGraphNpz(input_path)
        types = loadTypesNpz(input_path)
        domains = load_domains(input_path)
        ranges = load_ranges(input_path)
        type_hierarchy = load_type_hierarchy(input_path)
        prop_hierarchy = load_prop_hierarchy(input_path)
        types_dict = load_types_dict(input_path)
        rels_dict = load_relations_dict(input_path)

        if not isinstance(types, csr_matrix):
            types = types.tocsr()

        logger.info("learning types distributions")
        count_entities = types.shape[0]
        count_types = types.shape[1]
        count_relations = len(X)
        count_facts = sum([Xi.nnz for Xi in X])

        dist_types = {}
        for type in types:
            e_types = MultiType(type.indices)
            if e_types not in dist_types:
                dist_types[e_types] = 0.0
            dist_types[e_types] += 1

        logger.info("learning relations distributions")
        dist_relations = {p: X[p].nnz for p in range(len(X))}
        dist_domains_relation = {}
        dist_ranges_domain_relation = {}

        for p in range(len(X)):
            dist_domains_relation[p] = {}
            dist_ranges_domain_relation[p] = {}
            for i in range(X[p].nnz):
                s = X[p].row[i]
                o = X[p].col[i]
                s_types = MultiType(types[s].indices)
                o_types = MultiType(types[o].indices)

                if s_types not in dist_domains_relation[p]:
                    dist_domains_relation[p][s_types] = 0
                    dist_ranges_domain_relation[p][s_types] = {}
                if o_types not in dist_ranges_domain_relation[p][s_types]:
                    dist_ranges_domain_relation[p][s_types][o_types] = 0

                dist_domains_relation[p][s_types] += 1
                dist_ranges_domain_relation[p][s_types][o_types] += 1

        naive_model = KBModelM1(type_hierarchy, prop_hierarchy, domains, ranges, count_entities, count_relations,
                                count_facts, count_types,
                                dist_types, dist_relations, dist_domains_relation, dist_ranges_domain_relation,
                                rels_dict, types_dict)

        return naive_model
