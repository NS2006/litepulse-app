"""
Local static server for the site with correct MIME types.

    python verify/serve.py [port]        # then open http://localhost:8765

Python's built-in http.server reads MIME types from the Windows registry,
which can serve .js as text/plain and break ES modules. The camera also needs
a secure context: localhost counts, a LAN IP does not (use HTTPS hosting or
`adb reverse tcp:8765 tcp:8765` to test on an Android phone).
"""

import functools
import http.server
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".task": "application/octet-stream",
        ".apk": "application/vnd.android.package-archive",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    handler = functools.partial(Handler, directory=ROOT)
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler) as server:
        print(f"Serving {ROOT} at http://localhost:{PORT}")
        server.serve_forever()
