import webbrowser
import threading
import socket
from main import create_app


def find_free_port(start=7860, end=7880):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return start


if __name__ == "__main__":
    port = find_free_port()
    url = f"http://localhost:{port}"
    app = create_app()
    print(f"Starting app — open your browser at: {url}")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.launch(share=False, server_name="0.0.0.0", server_port=port)
