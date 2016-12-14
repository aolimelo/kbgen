import re
from rdflib import URIRef
from util import URIRelation
from numpy import std, mean

nan = float("nan")


class Literal(object):
    def __init__(self, relation, arg1, arg2, rel_dict=None):
        self.arg1 = arg1
        self.arg2 = arg2
        self.relation = URIRelation(relation)
        self.rel_dict = None
        if rel_dict is not None:
            self.rel_dict = {v:k for k,v in rel_dict.items()}

    def __str__(self):
        if self.rel_dict is None or self.relation.id not in self.rel_dict:
            return "?"+chr(self.arg1+96)+"  "+str(self.relation.id)+"  ?"+chr(self.arg2+96)
        else:
            return "?"+chr(self.arg1+96)+"  "+self.rel_dict[self.relation.id]+"  ?"+chr(self.arg2+96)

    def sparql_patterns(self):
        return "?"+chr(self.arg1+96)+\
               " <"+self.relation.__str__()+"> " \
               "?"+chr(self.arg2+96)+" . "

    @staticmethod
    def parse_amie(literal_string, rel_dict):
        literal_string = literal_string.strip()
        args = literal_string.split(" ")
        args = filter(None,args)
        assert len(args)==3
        arg1 = args[0]
        arg2 = args[2]
        assert arg1.startswith("?") and arg2.startswith("?") and len(arg1)==2 and len(arg2)==2

        if args[1].startswith("<http") or args[1].startswith("http"):
            rel_uri = re.match("<?(.+)>?", args[1]).group(1)
            rel = URIRef(rel_uri)
        elif "dbo:" in args[1]:
            rel_name = re.match("<?dbo:(.+)>?", args[1]).group(1)
            rel = URIRef("http://dbpedia.org/ontology/"+rel_name)
        else:
            rel_name = re.match("<?(.+)>?", args[1]).group(1)
            rel = URIRef("http://wikidata.org/ontology/"+rel_name)

        if rel_dict is not None:
            if rel not in rel_dict:
                return None
        else:
            rel_dict = {rel:0}

        rel_id = rel_dict[rel]
        arg1_id = ord(arg1[1]) - 96
        arg2_id = ord(arg2[1]) - 96

        return Literal(rel_id,arg1_id,arg2_id,rel_dict)


# Rules are assumed to have their consequent always with first argument ?a and second ?b
class Rule(object):
    def __init__(self, antecedents=[], consequents=[], std_conf=1.0, pca_conf=1.0):
        self.antecedents = antecedents
        self.consequents = consequents
        self.std_conf = std_conf
        self.pca_conf = pca_conf

    def __str__(self):
        rule_str = ""
        for ant in self.antecedents:
            rule_str += ant.__str__() + "  "
        rule_str += "=>  "
        for con in self.consequents:
            rule_str += con.__str__() + "  "
        return rule_str

    def antecedents_sparql(self):
        patterns = ""
        for ant in self.antecedents:
            patterns += ant.sparql_patterns() + " . "
        return patterns

    def antecedents_patterns(self, g, s,p,o):
        patterns = ""
        arg_s = None
        arg_o = None
        new_lit = None
        p_uri = URIRef(p)
        for ant in self.antecedents:
            ant_p_uri = ant.relation.uri
            if ant_p_uri == p_uri:
                arg_s = "?"+chr(ant.arg1+96)
                arg_o = "?"+chr(ant.arg2+96)
                new_lit = ant
                break

        for ant in self.antecedents:
            if ant.relation != p:
                patterns += ant.sparql_patterns()

        s_ent = "<"+s+">"
        o_ent = "<"+o+">"

        patterns = patterns.replace(arg_s, s_ent) if arg_s is not None else patterns
        patterns = patterns.replace(arg_o, o_ent) if arg_s is not None else patterns

        return patterns, new_lit

    def produce(self, g, s,p,o):
        ss, ps, os = [],[],[]
        if len(self.antecedents) == 1:
            r_j = self.consequents[0].relation
            if isinstance(r_j, URIRelation):
                new_p = r_j.uri
            else:
                new_p = URIRelation(r_j).uri

            if self.antecedents[0].arg1 == self.consequents[0].arg1 and \
               self.antecedents[0].arg2 == self.consequents[0].arg2:
                ss.append(s), ps.append(new_p), os.append(o)

            if self.antecedents[0].arg1 == self.consequents[0].arg2 and \
               self.antecedents[0].arg2 == self.consequents[0].arg1:
                ss.append(o), ps.append(new_p), os.append(s)
            return zip(ss,ps,os)
        else:
            patterns, new_lit = self.antecedents_patterns(g,s,p,o)


            if "?b" not in patterns and "?a" not in patterns:
                projection = "ask "
            else:
                projection = "select where "
                if "?b" in patterns:
                    projection = projection.replace("select ", "select ?b ")
                if "?a" in patterns:
                    projection = projection.replace("select ", "select ?a ")

            patterns = "{"+patterns+"}"

            sparql_query = projection + patterns

            qres = g.query(sparql_query)

            r_j = self.consequents[0].relation
            if isinstance(r_j, URIRelation):
                new_p = self.consequents[0].relation.uri
            else:
                new_p = URIRelation(self.consequents[0].relation).uri

            if "?a" in projection and "?b" in projection:
                for a,b in qres:
                    new_s = a
                    new_o = b
                    ss.append(new_s), ps.append(new_p), os.append(new_o)
            elif "?a" in projection:
                new_o = s if new_lit.arg1==2 else o
                for a in qres:
                    new_s = a[0]
                    ss.append(new_s), ps.append(new_p), os.append(new_o)
            elif "?b" in projection:
                new_s = s if new_lit.arg1==1 else o
                for b in qres:
                    new_o = b[0]
                    ss.append(new_s), ps.append(new_p), os.append(new_o)
            else:
                if bool(qres):
                    new_s = s if new_lit.arg1==1 else o
                    new_o = o if new_lit.arg2==2 else s
                    ss.append(new_s), ps.append(new_p), os.append(new_o)

            return zip(ss,ps,os)



    @staticmethod
    def parse_amie(line, rel_dict):
        cells = line.split("\t")
        rule_string = cells[0]
        std_conf = float(cells[2].strip())
        pca_conf = float(cells[3].strip())
        assert "=>" in rule_string
        ant_cons = rule_string.split("=>")
        ant_cons = filter(None,ant_cons)
        ant_string = ant_cons[0].strip()
        con_string = ant_cons[1].strip()

        ant_string = re.sub("(\?\w+)\s+\?","\g<1>|?",ant_string)
        con_string = re.sub("(\?\w+)\s+\?","\g<1>|?",con_string)

        antecedents = []
        for ant in ant_string.split("|"):
            lit = Literal.parse_amie(ant,rel_dict)
            if lit is None:
                return None
            antecedents.append(lit)

        consequents = []
        for con in con_string.split("|"):
            lit = Literal.parse_amie(con,rel_dict)
            if lit is None:
                return None
            consequents.append(lit)

        return Rule(antecedents,consequents,std_conf,pca_conf)


class RuleSet(object):
    def __init__(self, rules=[]):
        self.rules = rules
        self.rules_per_relation = {}
        for rule in rules:
            for literal in rule.antecedents:
                if literal.relation.id not in self.rules_per_relation:
                    self.rules_per_relation[literal.relation.id] = []
                self.rules_per_relation[literal.relation.id].append(rule)


    @staticmethod
    def parse_amie(rules_path, rel_dict):
        f = open(rules_path,"rb")
        rules = []
        lines = f.readlines()
        for i in range(1,len(lines)):
            line = lines[i]
            if line.startswith("?"):
                rule = Rule.parse_amie(line,rel_dict)
                if rule is not None:
                    rules.append(rule)
        print("rules successfully parsed: %d"%len(rules))
        return RuleSet(rules)





class RuleStats(Rule):
    def __init__(self, antecedents=[], consequents=[], head_cov=nan, std_conf=nan, pca_conf=nan, pos_expl=nan,
                 std_body_sz=nan, pca_body_sz=nan, func_var=nan, std_low_bd=nan, pca_low_bd=nan, pca_conf_est=nan):
        super(RuleStats,self).__init__(antecedents,consequents,std_conf,pca_conf)
        self.head_cov = head_cov
        self.pos_expl = pos_expl
        self.std_body_sz = std_body_sz
        self.pca_body_sz = pca_body_sz
        self.func_var = func_var
        self.std_low_bd = std_low_bd
        self.pca_low_bd = pca_low_bd
        self.pca_conf_est = pca_conf_est

    @staticmethod
    def parse_amie(line, rel_dict):
        cells = line.split("\t")
        rule_string = cells[0]
        head_cov    = float(cells[1].strip())
        std_conf    = float(cells[2].strip())
        pca_conf    = float(cells[3].strip())
        pos_expl    = float(cells[4].strip())
        std_body_sz = float(cells[5].strip())
        pca_body_sz = float(cells[6].strip())
        #func_var = float(cells[7].strip())
        func_var = 0.0
        std_low_bd  = float(cells[8].strip())
        pca_low_bd  = float(cells[9].strip())
        pca_conf_est= float(cells[10].strip())
        assert "=>" in rule_string
        ant_cons = rule_string.split("=>")
        ant_cons = filter(None,ant_cons)
        ant_string = ant_cons[0].strip()
        con_string = ant_cons[1].strip()

        ant_string = re.sub("(\?\w+)\s+\?","\g<1>|?",ant_string)
        con_string = re.sub("(\?\w+)\s+\?","\g<1>|?",con_string)

        antecedents = []
        for ant in ant_string.split("|"):
            lit = Literal.parse_amie(ant,rel_dict)
            if lit is None:
                return None
            antecedents.append(lit)

        consequents = []
        for con in con_string.split("|"):
            lit = Literal.parse_amie(con,rel_dict)
            if lit is None:
                return None
            consequents.append(lit)

        return RuleStats(antecedents,consequents,head_cov,std_conf,pca_conf,pos_expl,
                         std_body_sz,pca_body_sz,func_var,std_low_bd,pca_low_bd,pca_conf_est)

class RuleSetStats(RuleSet):
    def __init__(self, rules=[]):
        super(RuleSetStats,self).__init__(rules)


    def avg(self,attname):
        values = []
        for rule in self.rules:
            values.append(rule.__getattribute__(attname))
        return mean(values), std(values)


    @staticmethod
    def parse_amie(rules_path, rel_dict):
        f = open(rules_path,"rb")
        rules = []
        lines = f.readlines()
        for i in range(1,len(lines)):
            line = lines[i]
            if line.startswith("?"):
                rule = RuleStats.parse_amie(line,rel_dict)
                if rule is not None:
                    rules.append(rule)
        return RuleSetStats(rules)




