#!/usr/bin/env python3
"""A tiny terminal chat with the local server. Standard library only.

  python3 chat.py                 interactive; type /quit to leave
  python3 chat.py --once "Hi"     one question, then exit
"""
import argparse, json, sys, time, urllib.error, urllib.request


def ask(base, messages, tokens):
    body = {"messages": messages, "max_tokens": tokens, "stream": True}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    reply, n, t0 = "", 0, time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta", {})
            if d.get("reasoning_content"):
                sys.stderr.write(d["reasoning_content"]); sys.stderr.flush()
            if d.get("content"):
                print(d["content"], end="", flush=True)
                reply += d["content"]
            if d.get("content") or d.get("reasoning_content"):
                n += 1
    print()
    sys.stderr.write(f"[{n} chunks in {time.time() - t0:.1f} s]\n")
    return reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--once", default=None)
    ap.add_argument("--file", default=None, help="text file added after the question (with --once)")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    if a.once is not None:
        text = a.once
        if a.file:
            text += "\n\n" + open(a.file, errors="replace").read()[:12000]
        ask(base, [{"role": "user", "content": text}], a.tokens)
        return
    history = []
    print("Type your message. /quit to leave, /new to start over.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if line == "/quit":
            return
        if line == "/new":
            history = []; continue
        if not line:
            continue
        history.append({"role": "user", "content": line})
        history.append({"role": "assistant", "content": ask(base, history, a.tokens)})


if __name__ == "__main__":
    try:
        main()
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise SystemExit(f"Could not reach the server ({e}). Is it running? Start it with ./run.sh serve")
