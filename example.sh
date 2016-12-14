#!/usr/bin/env bash

wget http://data.dws.informatik.uni-mannheim.de/hmctp/kbgen/eswc2015.n3
wget http://data.dws.informatik.uni-mannheim.de/hmctp/kbgen/eswc2015-AmieRules.txt

python load_tensor.py eswc2015.n3

python learn_model.py eswc2015.npz -m M1
python learn_model.py eswc2015.npz -m M2
python learn_model.py eswc2015.npz -m M3 -r eswc2015-AmieRules.txt
python learn_model.py eswc2015.npz -m e -sm M1 M2 M3

for m in M1 M2 M3 eM1 eM2 eM3; do
    python synthesize.py eswc2015-${m}.pkl eswc2015-replica-${m}.n3 -s 0.1
done