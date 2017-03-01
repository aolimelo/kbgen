from rdflib import URIRef
import codecs
import numpy as np
import logging
from copy import deepcopy
from load_tensor_tools import get_roots


def create_logger(level=logging.INFO, name="kbgen"):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    #console_handler = logging.StreamHandler()
    #console_handler.setLevel(level)
    #logger.addHandler(console_handler)
    file_handler = logging.FileHandler("%s.log"%name,mode='w')
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger


def level_hierarchy(hier):
    if hier is None:
        return []

    roots = get_roots(hier)
    remaining = deepcopy(hier.keys())
    level = roots
    levels = []
    while level:
        next_level = []
        for n in level:
            for c in n.children:
                if c.node_id in remaining:
                    next_level.append(c)
                    remaining.remove(c.node_id)

        levels.append(level)
        level = next_level

    return levels


class URIEntity(object):
    prefix = "http://dws.uni-mannheim.de/synthesized/Entity_"

    def __init__(self, r):
        self.uri = URIRef(self.prefix + str(r))
        self.id = r

    def __eq__(self, other):
        return isinstance(other,self.__class__) and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.uri.__str__()

    @staticmethod
    def extract_id(uri):
        if type(uri) == URIRef or type(uri) == URIEntity:
            uri = uri.__str__()
        assert uri.startswith(URIEntity.prefix)
        id = int(uri[uri.rindex("_")+1:])
        return URIEntity(id)


class URIRelation(URIEntity):
    prefix = "http://dws.uni-mannheim.de/synthesized/relation_"

    def __init__(self,r):
        super(URIRelation,self).__init__(r)

    @staticmethod
    def extract_id(uri):
        if type(uri) == URIRef or type(uri) == URIRelation:
            uri = uri.__str__()
        assert uri.startswith(URIRelation.prefix)
        id = int(uri[uri.rindex("_")+1:])
        return URIRelation(id)


class URIType(URIEntity):
    prefix = "http://dws.uni-mannheim.de/synthesized/Type_"

    def __init__(self,r):
        super(URIType,self).__init__(r)

    @staticmethod
    def extract_id(uri):
        if type(uri) == URIRef or type(uri) == URIType:
            uri = uri.__str__()
        assert uri.startswith(URIType.prefix)
        id = int(uri[uri.rindex("_")+1:])
        return URIType(id)


class MultiType(object):
    def __init__(self, types_list):
        self.types = set(types_list)

    def __eq__(self, other):
        return self.types == other.types

    def __hash__(self):
        h = 1
        for t in self.types:
            h *= t
        return int(h)

    def __str__(self):
        return str(self.types)


def dump_tsv(g,output_file):
    f = codecs.open(output_file,"wb",encoding="utf-8")
    for s,p,o in g:
        f.write(s+"\t"+p+"\t"+o+"\n")
    f.close()


def normalize(l):
    l_array = np.array(l).astype(float)
    l_array /= sum(l_array)
    return l_array.tolist()
