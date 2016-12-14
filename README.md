# Generation of artificial knowledge graphs

Implementation of synthesis model from the paper "Synthesizing Knowledge Graphs for Link and Type Prediction Benchmarking" submitted to ESWC2017.


## Knowledge base models

The six models described in the paper are supported:

- M1: Includes distribution of types over entities plus joint distribuiton of relations, subject and object types
- M2: M1 plus non-reflexiveness, functionality and inverse functionality of relations
- M3: M2 plus horn rules
- EM*i*: M*i* plus bias to selection of entities

## Usage

For this example we use a small knowledge base from the Semantic Web dog food about the conference
[`ESWC2015`](http://data.semanticweb.org/dumps/conferences/eswc-2015-complete.rdf).

- First load the knowledge base into a tensor:

 ```
 python load_tensor.py eswc-2015-complete.rdf
 ```

 this will create the file eswc-2015-complete.npz with a tensor representation of the knowledge base

- Then Learn the models:

 ```
 python learn_model.py eswc-2015-complete.ext -m M1
 python learn_model.py eswc-2015-complete.ext -m M2
 python learn_model.py eswc-2015-complete.ext -m M3 -r <path-to-amie-rules-file>
 python learn_model.py eswc-2015-complete.ext -m e -sm M1 M2 M3
 ```

 The commands need to be executed in the order above because one model is an extension of the other.
 The commands will create generate the models pickle files

 - ```eswc-2015-complete-M1.pkl```
 - ```eswc-2015-complete-M2.pkl```
 - ```eswc-2015-complete-M3.pkl```
 - ```eswc-2015-complete-eM1.pkl```, ```eswc-2015-complete-eM2.pkl``` and ```eswc-2015-complete-eM3.pkl```

- From the learned models the knowledge base can be synthesized

 ```
 python synthesize.py eswc-2015-complete-M1.pkl eswc-2015-replica-M1.n3 -size 0.1
 ```

 This will synthesize a replica of the dataset with 10% of the original size and dump it into ```eswc-2015-replica-M1.n3```


## Requirements
The knowledge base creation is done with [`rdflib`](https://github.com/RDFLib/rdflib).
The knowledge graph model M3 requires a text file containing horn rules learned with
[`AMIE`](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/amie/)