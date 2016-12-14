from rdflib import Graph
import numpy as np
from rdflib.namespace import RDF, RDFS, OWL
from argparse import ArgumentParser
from load_tensor_tools import load_type_dict, get_ranges, get_domains, get_type_dag, get_prop_dag
from scipy.sparse import lil_matrix, coo_matrix


parser = ArgumentParser(description="Load tensor from rdf data")
parser.add_argument("input", type=str, default=None, help="path to dataset on which to perform the evaluation")
parser.add_argument("-td","--typedict", type=str, default=None, help="path to types dictionary (tensor from which the synthesized data was loaded from)")

args = parser.parse_args()

print(args)


rdf_format = args.input[args.input.rindex(".")+1:]

print("loading data")
g =Graph()
g.parse(args.input,format=rdf_format)

dict_s = {}
dict_p = {}
dict_t = {}

print("loading types")
type_i = 0
for s,p,o in g.triples((None,RDF.type,None)):
    if o not in dict_t:
        dict_t[o] = type_i
        type_i += 1
for s,p,o in g.triples((None,RDF.type,OWL.Class)):
    if s not in dict_t:
        dict_t[s] = type_i
        type_i += 1
for s,p,o in g.triples((None,RDFS.subClassOf,None)):
    if s not in dict_t:
        dict_t[s] = type_i
        type_i += 1
    if o not in dict_t:
        dict_t[o] = type_i
        type_i += 1
print(str(len(dict_t)) + " types loaded")

print("loading subjects dict")
s_i = 0
for s in g.subjects():
    if s not in dict_s and s not in dict_t:
        dict_s[s] = s_i
        s_i += 1

print("loading object properties")
p_i = 0
for s,p,o in g.triples((None,None,None)):
    if p not in dict_p:
        if s in dict_s and o in dict_s:
            dict_p[p] = p_i
            p_i += 1
for s,p,o in g.triples((None,OWL.ObjectProperty,None)):
    if s not in dict_p:
        dict_p[p] = p_i
        p_i += 1
print(str(len(dict_p))+" object properties loaded")


print("allocating adjacency matrices")
data_coo = [{"rows":[],"cols":[],"vals":[]} for i in range(len(dict_p))]

print("populating adjacency matrices")

for s,p,o in g.triples((None,None,None)):
    if s in dict_s and o in dict_s and p in dict_p:
        s_i = dict_s[s]
        o_i = dict_s[o]
        p_i = dict_p[p]
        data_coo[p_i]["rows"].append(s_i)
        data_coo[p_i]["cols"].append(o_i)
        data_coo[p_i]["vals"].append(1)
        #data[p_i][s_i,o_i] = 1

data = [coo_matrix((p["vals"], (p["rows"],p["cols"])), shape=(len(dict_s),len(dict_s))) for p in data_coo]
data_coo = None

type_coo = {"rows":[],"cols":[],"vals":[]}
print("populating type matrix with type assertions")
val = 1.0
type_assertions = 0
for s,p,o in g.triples((None,RDF.type,None)):
    if s in dict_s and o in dict_t:
        s_i = dict_s[s]
        o_i = dict_t[o]
        type_coo["rows"].append(s_i)
        type_coo["cols"].append(o_i)
        type_coo["vals"].append(1)
        type_assertions += 1

typedata = coo_matrix((type_coo["vals"],(type_coo["rows"],type_coo["cols"])),shape=(len(dict_s),len(dict_t)),dtype=int)
print(str(type_assertions)+" type assertions loaded")

n_labels = len(dict_t)

type_hierarchy = get_type_dag(g, dict_t)
prop_hierarchy = get_prop_dag(g, dict_p)

# change from objects to indices to avoid "maximum recursion depth exceeded" when pickling
for i,n in type_hierarchy.items():
    n.children = [c.node_id for c in n.children]
    n.parents = [p.node_id for p in n.parents]
for i,n in prop_hierarchy.items():
    n.children = [c.node_id for c in n.children]
    n.parents = [p.node_id for p in n.parents]

type_total = len(dict_t) if dict_t else 0
type_matched = len(type_hierarchy) if type_hierarchy else 0
prop_total = len(dict_t) if dict_t else 0
prop_matched = len(prop_hierarchy) if prop_hierarchy else 0

print("load types hierarchy: total=%d matched=%d"%(type_total,type_matched))
print("load relations hierarchy: total=%d matched=%d"%(prop_total,prop_matched))

domains = get_domains(g,dict_p,dict_t)
print("load relation domains: total=%d"%(len(domains)))

ranges = get_ranges(g,dict_p,dict_t)
print("load relation ranges: total=%d" % (len(ranges)))


rdfs = {"type_hierarchy": type_hierarchy,
        "prop_hierarchy": prop_hierarchy,
        "domains":domains,
        "ranges":ranges}

np.savez(args.input.replace("."+rdf_format, "-ext.npz"),
        data=data,
        types=typedata,
        entities_dict=dict_s,
        relations_dict=dict_p,
        types_dict=dict_t,
        type_hierarchy=type_hierarchy,
        prop_hierarchy=prop_hierarchy,
        domains=domains,
        ranges=ranges)