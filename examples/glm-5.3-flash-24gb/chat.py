#!/usr/bin/env python3
"""A tiny terminal chat with the local server. Standard library only.

  python3 chat.py                 interactive; type /quit to leave
  python3 chat.py --once "Hi"     one question, then exit
  python3 chat.py --think         also show the model's thinking (dimmed) before each answer
"""
import argparse, json, os, sys, time, urllib.error, urllib.request


DIM, PLAIN = "\033[2m", "\033[0m"


def ask(base, messages, tokens, think=False, quiet=False):
    """Streams one reply. Shows the answer only; the thinking too with think; progress dots only with quiet."""
    body = {"messages": messages, "max_tokens": tokens, "stream": True}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    colour = sys.stderr.isatty()
    reply, n, thinking, told, t0 = "", 0, False, False, time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta", {})
            if not (d.get("content") or d.get("reasoning_content")):
                continue
            n += 1
            if quiet:
                if n % 20 == 0:
                    sys.stderr.write("."); sys.stderr.flush()
                continue
            if d.get("reasoning_content") and not think and not told:
                sys.stderr.write("(thinking"); told = True
            if d.get("reasoning_content") and not think and n % 20 == 0:
                sys.stderr.write("."); sys.stderr.flush()
            if d.get("reasoning_content") and think:
                if not thinking:
                    sys.stderr.write((DIM if colour else "") + "[thinking] "); thinking = True
                sys.stderr.write(d["reasoning_content"]); sys.stderr.flush()
            if d.get("content"):
                if told:
                    sys.stderr.write(")\n"); told = False
                if thinking:
                    sys.stderr.write((PLAIN if colour else "") + "\n"); sys.stderr.flush(); thinking = False
                print(d["content"], end="", flush=True)
                reply += d["content"]
    if told:
        sys.stderr.write(")\n")
    if thinking:
        sys.stderr.write((PLAIN if colour else "") + "\n")
    if quiet:
        if n >= 20:
            sys.stderr.write("\n")
    else:
        print(flush=True)
        if n and not reply and not think:
            sys.stderr.write("(No answer: the reply ran out of tokens while the model was still thinking. --think or /think shows it; --tokens N allows more.)\n")
    sys.stderr.write(f"[{n} chunks in {time.time() - t0:.1f} s]\n")
    return reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--once", default=None)
    ap.add_argument("--file", default=None, help="text file added after the question (with --once)")
    ap.add_argument("--think", action="store_true", help="also show the model's thinking, dimmed, before the answer")
    ap.add_argument("--quiet", action="store_true", help="print progress dots, not the reply (with --once)")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"
    if a.once is not None:
        text = a.once
        if a.file:
            if not os.path.isfile(a.file):
                raise SystemExit(f"{a.file} is not a file, so there is no text to add.")
            with open(a.file, errors="replace") as f:
                extra = f.read(12000)
            if not extra.strip():
                raise SystemExit(f"{a.file} is empty, so there is no text to add.")
            text += "\n\n" + extra
        ask(base, [{"role": "user", "content": text}], a.tokens, a.think, a.quiet)
        return
    history, think = [], a.think
    print("Type your message. /quit to leave, /new to start over, /think to show or hide the model's thinking.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if line == "/quit":
            return
        if line == "/new":
            history = []; print("Started over."); continue
        if line == "/think":
            think = not think; print("Thinking is shown." if think else "Thinking is hidden."); continue
        if not line:
            continue
        history.append({"role": "user", "content": line})
        reply = ask(base, history, a.tokens, think)
        if reply:
            history.append({"role": "assistant", "content": reply})
        else:
            history.pop()


if __name__ == "__main__":
    try:
        main()
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise SystemExit(f"Could not reach the server ({e}). Is it running? Start it with ./run.sh serve")
