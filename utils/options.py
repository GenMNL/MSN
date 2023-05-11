import argparse

# ----------------------------------------------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser(description="options of PCN")

    # make parser for train (part of this is used for test)
    parser.add_argument("--num_output_points", default=16384, type=int)
    parser.add_argument("--emb_dim", default=1024, type=int)
    parser.add_argument("--num_surfaces", default=32, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("-sm", "--sampling_method", default="random", help="You can use MDS if you use pytorch1.2.0 or FPS")
    parser.add_argument("--optimizer", default="Adam", help="if you want to choose other optimization, you must change the code.")
    parser.add_argument("--lr", default=1e-3, help="learning rate", type=float)
    parser.add_argument("--dataset_dir", default="./../../dataset/ShapeNetCompletion")
    parser.add_argument("--save_dir", default="./checkpoint")
    parser.add_argument("--subset", default="all")
    parser.add_argument("--device", default="cuda")

    # make parser for test
    parser.add_argument("--result_dir", default="./result")
    parser.add_argument("--select_result", default="best") # you can select best or normal
    parser.add_argument("--result_subset", default="all")
    parser.add_argument("--result_eval", default="test")
    parser.add_argument("--year", default="2023")
    parser.add_argument("-d", "--date", type=str)
    return parser
# ----------------------------------------------------------------------------------------
