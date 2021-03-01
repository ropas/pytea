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
from json2z3 import Z3Encoder, PathResult, CtrSet

DEFAULT_PORT = 17851


def poll_socket(socket, timetick=100):
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    # wait up to 100msec
    try:
        while True:
            obj = dict(poller.poll(timetick))
            if socket in obj and obj[socket] == zmq.POLLIN:
                yield socket.recv()
    except KeyboardInterrupt:
        pass


def respond_rpc(id, result):
    ret_val = dict()
    ret_val["jsonrpc"] = "2.0"
    ret_val["id"] = id
    ret_val["result"] = json.dumps(result)
    return ret_val


class PyTeaServer:
    def __init__(self, port):
        self.port = port
        self.encoder = Z3Encoder(self)

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://127.0.0.1:{port}")
        print(f"python server listening port {port}...")

    def listen(self):
        for raw_message in poll_socket(self.socket):
            raw_message = self.socket.recv()
            message = json.loads(raw_message)
            result = []
            message_id = message["id"]

            if message["method"] == "ping":
                result.append(respond_rpc(message_id, result))
            elif message["method"] == "solve":
                result = self.analyze(message["result"])

            self.socket.send(json.dumps(result).encode("utf-8"))

    def analyze(self, paths):
        ctr_set_list = map(CtrSet, paths)
        result = []

        for ctr_set in ctr_set_list:
            path_result = dict()
            path_type, _, extras = ctr_set.analysis()  # side effect: print result

            path_result["type"] = path_type
            path_result["extras"] = extras

            result.append(path_result)

        return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTea z3/html server. Communicates with node server by JSON-RPC"
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        required=True,
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

    server = PyTeaServer(args.port)
    server.listen()


if __name__ == "__main__":
    main()

