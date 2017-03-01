import pickle
from argparse import ArgumentParser

from kb_models.model_m3 import KBModelM3
from kb_models.model_m1 import KBModelM1
from kb_models.model_m2 import KBModelM2

from kb_models.model_emi import KBModelEMi
from rules import RuleSet

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input", type=str, default=None, help="path to the tensor npz file")
    parser.add_argument("-m", "--model", type=str, default="M1",
                        help="choice of model [M1, M2, M2, e] (e requires -sm)")
    parser.add_argument("-sm", "--source-kb-models", type=str, nargs="+", default=["M1", "M2", "M3"],
                        help="source model with entity selection bias [M1, M2, M3]")
    parser.add_argument("-r", "--rules-path", type=str, default=None, help="path to txt file with Amie horn rules")

    args = parser.parse_args()

    base = args.input.replace(".npz", "")

    print(args)
    print("learning " + args.model + " model")

    model_output = ""

    models = []
    models_output = []

    if args.model == "M1":
        models.append(KBModelM1.generate_from_tensor(args.input))
        models_output.append(base + "-M1.pkl")

    if args.model == "M2":
        m1_model_path = base + "-M1.pkl"
        m1_model = pickle.load(open(m1_model_path, "rb"))
        assert isinstance(m1_model, KBModelM1)
        models.append(KBModelM2.generate_from_tensor(m1_model, args.input))
        models_output.append(base + "-M2.pkl")

    if args.model == "M3":
        m2_model_path = base + "-M2.pkl"
        m2_model = pickle.load(open(m2_model_path, "rb"))
        assert isinstance(m2_model, KBModelM2)
        rel_dict = m2_model.rel_dict
        rules = RuleSet.parse_amie(args.rules_path, rel_dict)
        models.append(KBModelM3(m2_model, rules))
        models_output.append(base + "-M3.pkl")

    if args.model == "e":
        dist_subjects, dist_objects = None, None
        for source_model_name in args.source_kb_models:
            m1_model_path = base + "-" + source_model_name + ".pkl"
            m1_model = pickle.load(open(m1_model_path, "rb"))
            assert isinstance(m1_model, KBModelM1)
            if dist_subjects is None and dist_objects is None:
                model = KBModelEMi.generate_from_tensor(m1_model, args.input)
                dist_subjects = model.dist_subjects
                dist_objects = model.dist_objects
                models.append(model)
            else:
                models.append(KBModelEMi(m1_model, dist_subjects, dist_objects))
            models_output.append(base + "-e" + source_model_name + ".pkl")

    if models and models_output:
        for model, model_output in zip(models, models_output):
            print("saving model to " + model_output)
            pickle.dump(model, open(model_output, "wb"))
