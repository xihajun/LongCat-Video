"""
Stable public tunnels for Gradio on Kaggle.

localtunnel (`lt`) drops connections after minutes and rotates URLs.
This module provides two far more stable options:

1. Cloudflare quick tunnel (default, zero-config)
   - No account needed. Much more stable than localtunnel.
   - Supervised: if the tunnel process dies it is restarted automatically
     and the new URL is printed / written to a file.

2. ngrok with a free STATIC domain (recommended for 2h+ sessions)
   - Sign up at https://dashboard.ngrok.com (free), grab:
       * your authtoken:      https://dashboard.ngrok.com/get-started/your-authtoken
       * one free static domain: https://dashboard.ngrok.com/domains
   - The URL NEVER changes, even across restarts, so reconnecting always
     works with the same link.

Usage in a Kaggle notebook cell:

    from kaggle_tunnel import start_cloudflared, start_ngrok

    # Option A: zero-config
    start_cloudflared(7860)

    # Option B: fixed URL forever (best)
    start_ngrok(7860, authtoken="YOUR_TOKEN", domain="your-name.ngrok-free.app")
"""

import os
import re
import subprocess
import threading
import time
import urllib.request

URL_FILE = "/kaggle/working/tunnel_url.txt" if os.path.isdir("/kaggle/working") else "/tmp/tunnel_url.txt"
CLOUDFLARED_BIN = "/kaggle/working/cloudflared" if os.path.isdir("/kaggle/working") else "/tmp/cloudflared"

_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _write_url(url):
    with open(URL_FILE, "w") as f:
        f.write(url + "\n")
    print(f"\n{'='*60}\n  PUBLIC URL: {url}\n  (also saved to {URL_FILE})\n{'='*60}\n", flush=True)


def ensure_cloudflared():
    """Download the cloudflared binary once."""
    if os.path.exists(CLOUDFLARED_BIN):
        return CLOUDFLARED_BIN
    print("[tunnel] downloading cloudflared...", flush=True)
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    urllib.request.urlretrieve(url, CLOUDFLARED_BIN)
    os.chmod(CLOUDFLARED_BIN, 0o755)
    return CLOUDFLARED_BIN


def _wait_port(port, timeout=300):
    """Block until the local server answers (so the tunnel doesn't 502)."""
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(2)
    return False


def start_cloudflared(port=7860, log_path=None):
    """Start a supervised Cloudflare quick tunnel in a background thread.

    Restarts automatically if the process dies; prints the (new) URL each time.
    Returns the supervisor thread.
    """
    binary = ensure_cloudflared()
    log_path = log_path or ("/kaggle/working/cloudflared.log"
                            if os.path.isdir("/kaggle/working") else "/tmp/cloudflared.log")

    def supervisor():
        restart = 0
        while True:
            if not _wait_port(port):
                print(f"[tunnel] port {port} never came up, retrying...", flush=True)
                continue
            cmd = [
                binary, "tunnel",
                "--url", f"http://127.0.0.1:{port}",
                "--no-autoupdate",
                # keep the connection alive aggressively
                "--protocol", "http2",
                "--edge-ip-version", "auto",
            ]
            print(f"[tunnel] starting cloudflared (attempt {restart + 1})", flush=True)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            logf = open(log_path, "a")
            try:
                for line in proc.stdout:
                    logf.write(line)
                    logf.flush()
                    m = _URL_RE.search(line)
                    if m:
                        _write_url(m.group(0))
            finally:
                logf.close()
            code = proc.wait()
            restart += 1
            print(f"[tunnel] cloudflared exited (code={code}), restarting in 3s "
                  f"(restart #{restart})", flush=True)
            time.sleep(3)

    t = threading.Thread(target=supervisor, daemon=True, name="cloudflared-supervisor")
    t.start()
    return t


def start_ngrok(port=7860, authtoken=None, domain=None):
    """Start an ngrok tunnel via pyngrok. With a free static `domain`,
    the URL never changes across restarts/reconnects.

    Returns the public URL. A watchdog thread reconnects if the tunnel dies.
    """
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        subprocess.check_call(["pip", "install", "-q", "pyngrok"])
        from pyngrok import ngrok, conf

    if authtoken:
        ngrok.set_auth_token(authtoken)
    conf.get_default().monitor_thread = False

    _wait_port(port)

    def _connect():
        kwargs = {"proto": "http"}
        if domain:
            kwargs["domain"] = domain
        tunnel = ngrok.connect(port, **kwargs)
        _write_url(tunnel.public_url)
        return tunnel

    tunnel = _connect()

    def watchdog():
        while True:
            time.sleep(30)
            try:
                alive = any(t.proto in ("http", "https")
                            for t in ngrok.get_tunnels())
            except Exception:
                alive = False
            if not alive:
                print("[tunnel] ngrok tunnel lost, reconnecting...", flush=True)
                try:
                    _connect()
                except Exception as e:
                    print(f"[tunnel] reconnect failed: {e}", flush=True)

    threading.Thread(target=watchdog, daemon=True, name="ngrok-watchdog").start()
    return tunnel.public_url


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--backend", choices=["cloudflared", "ngrok"], default="cloudflared")
    p.add_argument("--authtoken", default=os.environ.get("NGROK_AUTHTOKEN"))
    p.add_argument("--domain", default=os.environ.get("NGROK_DOMAIN"))
    a = p.parse_args()
    if a.backend == "ngrok":
        start_ngrok(a.port, a.authtoken, a.domain)
        while True:
            time.sleep(3600)
    else:
        start_cloudflared(a.port).join()