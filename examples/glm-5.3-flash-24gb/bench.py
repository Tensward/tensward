#!/usr/bin/env python3
"""Speed check against a local llama-server. Standard library only.

Run it right after the server starts to see cold against warm speed.
"""
import argparse, json, random, re, statistics, time, urllib.error, urllib.request

WORDS = ("river stone garden window copper lantern meadow harbor ticket orchard "
         "violin pencil bridge market cloud timber ladder summer button kettle "
         "marble forest signal paper candle anchor valley engine carpet mirror").split()
STORY = "Write a long, calm story about a lighthouse keeper and the seasons."


def post(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(base, text, n, ignore_eos=False):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": n,
            "temperature": 0.0, "cache_prompt": True}
    if ignore_eos:
        body["ignore_eos"] = True
    t0 = time.time()
    out = post(base, "/v1/chat/completions", body)
    return out.get("timings", {}), time.time() - t0


def make_text(base, target):
    """A fresh neutral text of about `target` tokens (new every call, so no prompt-cache hit)."""
    seed = random.randrange(10**9)
    rnd = random.Random(seed)
    words = [rnd.choice(WORDS) for _ in range(int(target * 1.2))]
    for _ in range(4):
        text = f"Reference {seed}. " + " ".join(words)
        n = len(post(base, "/tokenize", {"content": text})["tokens"])
        if abs(n - target) <= max(20, target // 100):
            break
        if n > target:
            words = words[: max(1, int(len(words) * target / n))]
        else:
            words += [rnd.choice(WORDS) for _ in range(int((target - n) * 0.8) + 1)]
    return text


def prompt_test(base, target):
    t, wall = chat(base, make_text(base, target) + "\nSay OK.", 1)
    return {"tokens": t.get("prompt_n"), "seconds": round(t.get("prompt_ms", 0) / 1000, 2),
            "tok_per_s": round(t.get("prompt_per_second", 0.0), 2), "wall_seconds": round(wall, 2)}


def ttft(base):
    body = {"messages": [{"role": "user",
                          "content": f"Reference {random.randrange(10**9)}. Describe a quiet harbor."}],
            "max_tokens": 32, "stream": True}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta", {})
            if d.get("content") or d.get("reasoning_content"):
                return round(time.time() - t0, 3)
    return None


def acceptance(log):
    """The draft head's numbers from the last server log line that reports them, if any."""
    pat = re.compile(r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated")
    try:
        found = [m for m in map(pat.search, open(log, errors="replace")) if m]
    except OSError:
        return None
    if not found:
        return None
    share, accepted, drafted = found[-1].groups()
    return {"accepted_share": float(share), "accepted": int(accepted), "drafted": int(drafted)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--log", default=None, help="server log, to read the draft head's acceptance numbers")
    ap.add_argument("--quick", action="store_true", help="one short speed probe (about 10 seconds)")
    ap.add_argument("--peak-probe", type=int, default=0, metavar="TOKENS",
                    help="only send one long fresh prompt of this size (used to size the card cache)")
    a = ap.parse_args()
    base = f"http://{a.host}:{a.port}"

    if a.quick:
        chat(base, STORY, 16)
        t, _ = chat(base, STORY + f" ({random.randrange(10**6)})", 64, ignore_eos=True)
        print(f"Speed: about {t.get('predicted_per_second', 0.0):.1f} tokens per second while writing.")
        return

    if a.peak_probe:
        chat(base, STORY, 8)
        prompt_test(base, a.peak_probe)
        return

    print("First 64 tokens (cold: best right after the server starts)...")
    t, _ = chat(base, STORY, 64, ignore_eos=True)
    first64 = round(t.get("predicted_per_second", 0.0), 2)

    print("Warm-up (not counted)...")
    chat(base, STORY, 64)

    print("Three steady replies of 192 tokens...")
    steady = [round(chat(base, STORY, 192, ignore_eos=True)[0].get("predicted_per_second", 0.0), 2)
              for _ in range(3)]

    print("Fresh prompts of about 256 and 4,000 tokens...")
    p256 = prompt_test(base, 256)
    p4000 = prompt_test(base, 4000)

    print("Time to first token...")
    first = ttft(base)

    res = {"first_64_tok_per_s": first64, "steady_tok_per_s": steady,
           "steady_median_tok_per_s": statistics.median(steady),
           "prompt_256": p256, "prompt_4000": p4000, "time_to_first_token_s": first,
           "draft_head": acceptance(a.log) if a.log else None,
           "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\nResult")
    print("-" * 46)
    print(f"First 64 tokens (tok/s)   {first64}")
    print(f"Steady replies (tok/s)    {steady}")
    print(f"Steady median (tok/s)     {res['steady_median_tok_per_s']}")
    for name, p in (("256-token prompt", p256), ("4,000-token prompt", p4000)):
        print(f"{name:<26}{p['tokens']} tokens, {p['seconds']} s, {p['tok_per_s']} tok/s")
    print(f"Time to first token (s)   {first}")
    dh = res["draft_head"]
    if dh:
        print(f"Draft head: {dh['accepted_share'] * 100:.1f}% of drafted tokens accepted ({dh['accepted']} of {dh['drafted']})")
    print(f"Saved to {a.out}")


if __name__ == "__main__":
    try:
        main()
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise SystemExit(f"Could not reach the server ({e}). Is it running? Start it with ./run.sh serve")
