import argparse
import torch

# ----------------------------------------------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser(description="options of PCN")

    # make parser for train (part of this is used for test)
    parser.add_argument("--num_partial", default=4000, type=int)
    parser.add_argument("--num_comp", default=16384, type=int)
    parser.add_argument("--num_output_points", default=16384, type=int)
    parser.add_argument("--emb_dim", default=1024, type=int)
    parser.add_argument("--num_surfaces", default=1024, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--optimizer", default="Adam", help="if you want to choose other optimization, you must change the code.")
    parser.add_argument("--lr", default=1e-4, help="learning rate", type=float)
    parser.add_argument("--dataset_dir", default="../PCN/data/BridgeCompletion")
    parser.add_argument("--save_dir", default="./checkpoint")
    parser.add_argument("--subset", default="bridge")
    parser.add_argument("--device", default="cuda")

    # make parser for test
    parser.add_argument("--result_dir", default="./result")
    parser.add_argument("--select_result", default="best") # you can select best or normal
    parser.add_argument("--result_subset", default="bridge")
    parser.add_argument("--result_eval", default="test")
    return parser
# ----------------------------------------------------------------------------------------
