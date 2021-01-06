#!/usr/bin/env python

"""
json2z3.py
Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Ho Young Jhoo

Main starting point of Pytea analyzer
"""
import os
import subprocess
import argparse
from pathlib import Path
from z3wrapper.json2z3 import run_default


def parse_arg():
    """
    pytorch_pgm_path : <path_to_pytorch>/<pytorch_pgm_name>.py
    json_file        : <pytorch_pgm_name>_z3.json
    """
    parser = argparse.ArgumentParser("PyTeA: PyTorch Tensor shape Analyzer")
    parser.add_argument("path", help="PyTorch entry file path")
    parser.add_argument(
        "--out",
        default="./constraint.json",
        help="z3 json output path. (default: ./constraint.json)",
    )
    parser.add_argument(
        "--z3_only", action="store_true", help="run z3py on z3 json file <path>"
    )
    parser.add_argument(
        "--no_z3",
        action="store_true",
        help="do not run z3py and just produce z3 json file",
    )
    parser.add_argument(
        "--front_path",
        default="./index.js",
        help="path to constraint generator (index.js)",
    )
    parser.add_argument(
        "--node-arguments", default="", help="arguments for constraint generator"
    )

    return parser.parse_args()


def main():
    args = parse_arg()

    entry_path = Path(args.path)
    if not entry_path.exists():
        raise Exception(f"entry path {entry_path} does not exist")

    base_dir = entry_path.parent

    json_path = Path(args.out)
    if args.z3_only:
        json_path = entry_path

    # run frontend with given path, creates json formatted constraints.
    if not args.z3_only:
        if json_path.exists():
            os.remove(json_path)

        frontend_command = f"node {args.front_path} {entry_path} {args.node_arguments}"
        print(frontend_command)
        subprocess.call(frontend_command, shell=True)

    # run backend with json formatted constraints.
    if not args.no_z3:
        print("\n------------- z3 result -------------")
        run_default(json_path)


if __name__ == "__main__":
    main()
