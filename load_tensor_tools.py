import numpy as np
from rdflib import OWL, RDFS
from multiprocessing import Queue


def to_triples(X, order="pso"):
    h, t, r = [], [], []
    for i in range(len(X)):
        r.extend(np.full((X[i].nnz), i))
        h.extend(X[i].row.tolist())
        t.extend(X[i].col.tolist())
    if order == "spo":
        triples = zip(h, r, t)
    if order == "pso":
        triples = zip(r, h, t)
    if order == "sop":
        triples = zip(h, t, r)
    return np.array(triples)


def load_domains(inputDir):
    dataset = np.load(inputDir)
    return dataset["domains"].item() if "domains" in dataset else None


def load_ranges(inputDir):
    dataset = np.load(inputDir)
    return dataset["ranges"].item() if "ranges" in dataset else None


def load_type_hierarchy(inputDir):
    dataset = np.load(inputDir)
    hierarchy = dataset["type_hierarchy"].item() if "type_hierarchy" in dataset else None
    for i, n in hierarchy.items():
        try:
            n.children = [hierarchy[c] for c in n.children]
            n.parents = [hierarchy[p] for p in n.parents]
        except:
            pass
    return hierarchy


def load_prop_hierarchy(inputDir):
    dataset = np.load(inputDir)
    hierarchy = dataset["prop_hierarchy"].item() if "prop_hierarchy" in dataset else None
    for i, n in hierarchy.items():
        try:
            n.children = [hierarchy[c] for c in n.children]
            n.parents = [hierarchy[p] for p in n.parents]
        except:
            pass
    return hierarchy


def load_entities_dict(inputDir):
    dataset = np.load(inputDir)
    return dataset["entities_dict"].item() if "entities_dict" in dataset else None


def load_types_dict(inputDir):
    dataset = np.load(inputDir)
    return dataset["types_dict"].item() if "types_dict" in dataset else None


def load_relations_dict(inputDir):
    dataset = np.load(inputDir)
    return dataset["relations_dict"].item() if "relations_dict" in dataset else None


def loadGraphNpz(inputDir):
    dataset = np.load(inputDir)
    data = dataset["data"]
    return data.tolist()


def loadTypesNpz(inputDir):
    dataset = np.load(inputDir)
    return dataset["types"].item()


def get_prop_dag(g, dict_rel):
    prop_dag = {}
    for s, p, o in g.triples((None, RDFS.subPropertyOf, None)):
        if (s in dict_rel) and (o in dict_rel):

            s_id = dict_rel[s]
            o_id = dict_rel[o]
            if s_id != o_id:
                if o_id not in prop_dag:
                    prop_dag[o_id] = DAGNode(o_id, o, parents=[], children=[])
                if s_id not in prop_dag:
                    prop_dag[s_id] = DAGNode(s_id, s, parents=[], children=[])

                prop_dag[s_id].parents.append(prop_dag[o_id])
                prop_dag[o_id].children.append(prop_dag[s_id])

    return prop_dag


def get_prop_tree(g, dict_rel):
    prop_tree = {}
    for s, p, o in g.triples((None, RDFS.subPropertyOf, None)):
        if (s in dict_rel) and (o in dict_rel):
            s_id = dict_rel[s]
            o_id = dict_rel[o]
            if s_id != o_id:
                if o_id not in prop_tree:
                    prop_tree[o_id] = TreeNode(o_id, o, children=[])
                if s_id not in prop_tree:
                    prop_tree[s_id] = TreeNode(s_id, s, prop_tree[o_id], children=[])

                if prop_tree[s_id].parent is None:
                    prop_tree[s_id].parent = prop_tree[o_id]
                if prop_tree[s_id].parent == prop_tree[o_id]:
                    prop_tree[o_id].children.append(prop_tree[s_id])

    return prop_tree


def get_type_dag(g, dict_type):
    type_dag = {}

    # getting equivalent classes
    equi_classes = {}
    for s, p, o in g.triples((None, OWL.equivalentClass, None)):
        equi_classes[s] = o
        equi_classes[o] = s

    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        if (s in dict_type or (s in equi_classes and equi_classes[s] in dict_type)) and \
                (o in dict_type or (o in equi_classes and equi_classes[o] in dict_type)):

            if s not in dict_type:
                s = equi_classes[s]
            if o not in dict_type:
                o = equi_classes[o]

            s_id = dict_type[s]
            o_id = dict_type[o]
            if s_id != o_id:
                if o_id not in type_dag:
                    type_dag[o_id] = DAGNode(o_id, o, parents=[], children=[])
                if s_id not in type_dag:
                    type_dag[s_id] = DAGNode(s_id, s, parents=[], children=[])

                type_dag[s_id].parents.append(type_dag[o_id])
                type_dag[o_id].children.append(type_dag[s_id])

    return type_dag


def get_type_tree(g, dict_type):
    type_tree = {}

    # getting equivalent classes
    equi_classes = {}
    for s, p, o in g.triples((None, OWL.equivalentClass, None)):
        equi_classes[s] = o
        equi_classes[o] = s

    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        if (s in dict_type or (s in equi_classes and equi_classes[s] in dict_type)) and \
                (o in dict_type or (o in equi_classes and equi_classes[o] in dict_type)):

            if s not in dict_type:
                s = equi_classes[s]
            if o not in dict_type:
                o = equi_classes[o]

            s_id = dict_type[s]
            o_id = dict_type[o]
            if s_id != o_id:
                if o_id not in type_tree:
                    type_tree[o_id] = TreeNode(o_id, o, children=[])
                if s_id not in type_tree:
                    type_tree[s_id] = TreeNode(s_id, s, type_tree[o_id], children=[])

                if type_tree[s_id].parent is None:
                    type_tree[s_id].parent = type_tree[o_id]
                if type_tree[s_id].parent == type_tree[o_id]:
                    type_tree[o_id].children.append(type_tree[s_id])

    return type_tree


def get_domains(g, dict_rel, dict_type):
    domains = {}
    for s, p, o in g.triples((None, RDFS.domain, None)):
        if s in dict_rel and o in dict_type:
            domains[dict_rel[s]] = dict_type[o]
    return domains


def get_ranges(g, dict_rel, dict_type):
    ranges = {}
    for s, p, o in g.triples((None, RDFS.range, None)):
        if s in dict_rel and o in dict_type:
            ranges[dict_rel[s]] = dict_type[o]
    return ranges


def load_type_dict(input_path):
    dataset = np.load(input_path)
    dict_type = dataset["types_dict"]
    if not isinstance(dict_type, dict):
        dict_type = dict_type.item()
    return dict_type


def load_relations_dict(input_path):
    dataset = np.load(input_path)
    dict_rel = dataset["relations_dict"]
    if not isinstance(dict_rel, dict):
        dict_rel = dict_rel.item()
    return dict_rel


class TreeNode(object):
    def __init__(self, node_id, name, parent=None, children=[]):
        self.node_id = node_id
        self.name = name
        self.parent = parent
        self.children = children

    def __str__(self):
        return self.name.__str__()

    def print_tree(self, tab="", pool=None):
        print(tab + self.name)
        for child in self.children:
            if self != child and self == child.parent:
                child.print_tree(tab + "\t")

    def get_all_parents(self):
        parents = []
        nd = self
        while nd.parent is not None:
            parents.append(nd.parent)
            nd = nd.parent
        return parents

    def get_all_parent_ids(self):
        return [p.id for p in self.get_all_parents()]


class DAGNode(object):
    def __init__(self, node_id, name, parents=[], children=[]):
        self.node_id = node_id
        self.name = name
        self.parents = parents
        self.children = children

    def __str__(self):
        return self.name.__str__()

    def print_tree(self, tab="", pool=None):
        print(tab + self.name)
        for child in self.children:
            if self != child and self in child.parents:
                child.print_tree(tab + "\t")

    def get_all_parents(self):
        parents = set()
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            nd = queue.get()
            for p in nd.parents:
                if p not in parents:
                    parents.add(p)
                    queue.put(p)
        return parents

    def get_all_parent_ids(self):
        return [p.node_id for p in self.get_all_parents()]

    def to_tree(self):
        tree_node = TreeNode(self.node_id, self.name)
        tree_node.children = [c.to_tree for c in self.children]
        tree_node.parent = min(self.parents)
        return tree_node


def get_roots(hier):
    if not hier:
        return []
    else:
        roots = []
        for i, n in hier.items():
            if isinstance(n, DAGNode):
                if not n.parents:
                    roots.append(n)
            if isinstance(n, TreeNode):
                if n.parent is None:
                    roots.append(n)
        return roots


def dag_to_tree(dag):
    tree = {}
    for i, n in dag.items():
        tree[i] = TreeNode(n.node_id, n.name)
    for i, n in dag.items():
        tree[i].children = [tree[c.node_id] for c in n.children]
        tree[i].parent = None if not n.parents else tree[min(n.parents).node_id]
    for i, n in tree.items():
        for c in n.children:
            if c.parent != n:
                n.children.remove(c)
    return tree
