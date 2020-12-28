#!/usr/bin/env python

"""
usage: chmod +x pytea.py
       ./pytea.py <pytorch_pgm_path>
"""
import os
import subprocess
import argparse
from pathlib import Path


def parse_arg():
    """
    pytorch_pgm_path : <path_to_pytorch>/<pytorch_pgm_name>.py
    json_file        : <pytorch_pgm_name>_z3.json
    """
    parser = argparse.ArgumentParser("PyTeA: PyTorch Tensor shape Analyzer")
    parser.add_argument("path", help="PyTorch entry file path")
    parser.add_argument(
        "--out", default="", help="z3 json output path. (default: <path>_z3.json)"
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
        "--backend_path", default="./pylib/json2z3.py", help="path to json2z3.py"
    )

    return parser.parse_args()


def main():
    args = parse_arg()

    backend_path = Path(args.backend_path)
    entry_path = Path(args.path)

    if not entry_path.exists():
        raise Exception(f"entry path {entry_path} does not exist")
    if not backend_path.exists():
        raise Exception(f"json2z3.py path {backend_path} does not exist")

    base_dir = entry_path.parent

    if args.out == "":
        json_path = base_dir / f"{entry_path.stem}_z3.json"
    else:
        json_path = Path(args.out)

    if args.z3_only:
        json_path = entry_path

    # run frontend with given path, creates json formatted constraints.
    if not args.z3_only:
        if json_path.exists():
            os.remove(json_path)

        frontend_command = f"npm run test:torch {entry_path}"
        subprocess.call(frontend_command, shell=True)

    # run backend with json formatted constraints.
    if not args.no_z3:
        if json_path.exists():
            backend_command = f"python {backend_path} {json_path}"
            subprocess.call(backend_command, shell=True)


if __name__ == "__main__":
    main()
