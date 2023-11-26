import torch
from argparse import ArgumentParser, Namespace


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser() -> ArgumentParser:
    programName: str = "LPCV 2023 Tiny Espresso Net"
    authors: list[str] = ["Holden Babineaux, Joseph Fontenot"]

    prog: str = programName
    usage: str = f"This is the {programName}"
    description: str = f"This {programName} does create a single" + \
        " segmentation map of areal scenes of disaster environments" + \
        " captured by unmanned areal vehicles (UAVs)"
    epilog: str = f"This {programName} was created by {''.join(authors)}"

    return ArgumentParser(prog, usage, description, epilog)


def get_eval_args() -> Namespace:
    parser: ArgumentParser = get_parser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Filepath to an image to create a segmentation map of",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Filepath to the corresponding output segmentation map",
    )
    return parser.parse_args()


def get_solution_args() -> Namespace:
    parser: ArgumentParser = get_parser
    parser.add_argument(
        "-i",
        "--image_dir",
        required=True,
        help="Directory containing the generated segmentation maps",
    )
    parser.add_argument(
        "-g",
        "--gt",
        required=True,
        help="Directory containing the ground truth images to compare to",
    )
    return parser.parse_args()
