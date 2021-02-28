"""
pyteaserver.py

Copyright (c) Seoul National University
Licensed under the MIT license.
Author: Ho Young Jhoo

runs simple JSON-RPC server,
analyze constraints by Z3 and serve it to HTML.
"""

import argparse
import importlib.util
import json
import sys
import zmq
from .json2z3 import Z3encoder

DEFAULT_PORT = 17851


class PyTeaServer:
    def __init__(self, port):
        self.port = port
        self.encoder = Z3encoder(self)

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://127.0.0.1:{port}")

    def log(self, message):
        pass

    def analyze(self, message):
        try:
            jsonObj = json.loads(message)
            self.encoder.analyze(jsonObj)
        except Exception as e:
            # TODO: do something
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTea z3/html server. Communicates with node server by JSON-RPC"
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help="main port to communicate with PyTea node server",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args is None:
        sys.exit(-1)

    if (
        importlib.util.find_spec("z3") is None
        or importlib.util.find_spec("zmq") is None
    ):
        sys.exit(-1)

    server = PyTeaServer(args)


if __name__ == "__main__":
    main()

