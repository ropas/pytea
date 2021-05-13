import argparse
import importlib.util
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from http import HTTPStatus
from json2z3 import CtrSet, PathResult, timeout


DEFAULT_PORT = 17851


def respond_rpc(id, result, **kwargs):
    ret_val = dict()
    ret_val["jsonrpc"] = "2.0"
    ret_val["id"] = id
    ret_val["result"] = result
    for k, v in kwargs.items():
        ret_val[k] = v
    return ret_val


def gen_handler(req_handler):
    class RequestHandler(BaseHTTPRequestHandler):
        # Borrowing from https://gist.github.com/nitaku/10d0662536f37a087e1b
        def _set_headers(self):
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-type", "application/json")
            # Allow requests from any origin, so CORS policies don't
            # prevent local development.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("content-length"))
            message = json.loads(self.rfile.read(length))
            result = req_handler(message)
            self._set_headers()
            self.wfile.write(result)

        def do_OPTIONS(self):
            # Send allow-origin header for preflight POST XHRs.
            self.send_response(HTTPStatus.NO_CONTENT.value)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.end_headers()

    return RequestHandler


class LocalLogger:
    def __init__(self):
        self.cache = []

    def log(self, message):
        self.cache.append(message)

    def flush(self):
        ret_val = "\n".join(self.cache)
        self.cache.clear()
        return ret_val


class PyTeaServer:
    def __init__(self, port):
        self.port = port
        self.httpd = HTTPServer(("127.0.0.1", port), gen_handler(self.handler))

        print(f"python server listening port {port}...")
        self.httpd.serve_forever()

    def handler(self, message):
        result = []
        message_id = message["id"]

        if message["method"] == "ping":
            result = message["params"]
        elif message["method"] == "solve":
            result = self.analyze(message["params"])
        else:
            result = None

        return json.dumps(respond_rpc(message_id, result)).encode("utf-8")

    def analyze(self, obj_list):
        ctrset_list = list(map(CtrSet, obj_list))
        result = []

        for ctr_set in ctrset_list:
            # 5 seconds timeout
            result_obj = dict()
            analyze_tm = timeout(5)(ctr_set.analysis)
            try:
                (
                    path_result,
                    path_log,
                    extras,
                ) = analyze_tm()  # side effect: print result

                result_obj["type"] = path_result
                result_obj["extras"] = extras
            except TimeoutError:
                result_obj["type"] = PathResult.Timeout.value
                result_obj["extras"] = dict()

            result.append(result_obj)

        return result


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

    if importlib.util.find_spec("z3") is None:
        sys.exit(-1)

    server = PyTeaServer(args.port)


if __name__ == "__main__":
    main()

