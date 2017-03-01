import pickle
from argparse import ArgumentParser
from util import dump_tsv

if __name__ == '__main__':
    parser = ArgumentParser(description="Synthesizes a dataset from a given model and size")
    parser.add_argument("input",type=str,default=None,help="path to kb model")
    parser.add_argument("output",type=str,default=None,help="path to the synthetic rdf file")
    parser.add_argument("-s","--size",type=float,default=1,help="sample size as number of original facts divided by step")
    parser.add_argument("-ne","--nentities",type=int,default=None,help="number of entities")
    parser.add_argument("-nf","--nfacts",type=int,default=None,help="number of facts")
    parser.add_argument("-d","--debug",dest="debug",action="store_true",help="debug mode")
    parser.set_defaults(debug=False)


    args = parser.parse_args()

    print(args)

    model = pickle.load(open(args.input,"rb"))

    g = model.synthesize(size=args.size, ne=args.nentities, nf=args.nfacts, debug=args.debug)

    rdf_format = args.output[args.output.rindex(".")+1:]
    g.serialize(open(args.output,"wb"), format=rdf_format)

    dump_tsv(g,args.output.replace("."+rdf_format,".tsv"))

