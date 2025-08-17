#!/usr/bin/env python3
import argparse, random, time, re, json, os, csv, urllib.request
from dataclasses import dataclass, asdict
from typing import List, Optional
import matplotlib.pyplot as plt

# --- Ollama bridge ---
OLLAMA_AVAILABLE = True
try:
    from ollama import chat as ollama_chat
except Exception:
    OLLAMA_AVAILABLE = False


def ollama_chat_fallback(model: str, messages: list, options: Optional[dict] = None) -> str:
    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    if options:
        payload["options"] = options
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        j = json.loads(resp.read().decode("utf-8", "ignore"))
        return j.get("message", {}).get("content", "") or j.get("content", "") or j.get("response", "")


def normalize_response(r) -> str:
    if isinstance(r, dict):
        return r.get("message", {}).get("content") or r.get("content") or r.get("response", "")
    if isinstance(r, str):
        return r
    return str(r)


def ollama_chat_call(model: str, messages: list, options: Optional[dict] = None) -> str:
    if OLLAMA_AVAILABLE:
        r = ollama_chat(model=model, messages=messages, options=options, stream=False)
        return normalize_response(r)
    return ollama_chat_fallback(model, messages, options)


# --- Map building ---
@dataclass
class Step:
    idx: int
    options: List[str]
    correct: int


HEX_DIGITS = "0123456789ABCDEF"


def hex_digit_at(idx: int) -> str:
    return HEX_DIGITS[(idx - 1) % len(HEX_DIGITS)]


def make_hex_sequence(length: int) -> List[str]:
    return [hex_digit_at(i) for i in range(1, length + 1)]


def make_step(idx: int, seq: List[str]) -> Step:
    key = seq[(idx - 1) % len(seq)]
    correct_slot = random.randint(1, 3)
    opts = []
    for slot in range(1, 4):
        digit = key if slot == correct_slot else random.choice(HEX_DIGITS.replace(key, ""))
        suffix = random.choice(["eep ravine", "usty barn", "ry well", "ark tunnel", "target"])
        opts.append(f"{digit} {suffix}")
    return Step(idx=idx, options=opts, correct=correct_slot)


def make_map(steps: int, seq: List[str], seed: Optional[int]) -> List[Step]:
    if seed is not None:
        random.seed(seed)
    return [make_step(i, seq) for i in range(1, steps + 1)]


# --- Prompt & parse ---
SYSTEM_INSTRUCTIONS = (
    "You are a MENU NAVIGATOR. Rules:\n"
    "1) You get a step number and 3 options.\n"
    "2) At step k, choose the option whose LABEL starts with the correct hex digit.\n"
    "3) Reply ONLY with 1, 2, or 3. No words.\n"
)

USER_TEMPLATE = (
    "Step {idx}\n"
    "Sequence: {seq}  (use '{key}')\n"
    "OPTIONS:\n"
    "1) {o1}\n"
    "2) {o2}\n"
    "3) {o3}\n\n"
    "Answer with ONLY 1, 2, or 3."
)

STRICT_OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.1,
    "mirostat": 0,
    "num_ctx": 2048,
}


def parse_choice(text: str) -> Optional[int]:
    m = re.search(r"\b([123])\b", text.strip())
    if m:
        return int(m.group(1))
    m = re.search(r"([123])", text)
    return int(m.group(1)) if m else None


def ask_model(model: str, idx: int, step: Step, seq: List[str], verbose=False):
    key = seq[(idx - 1) % len(seq)]
    user = USER_TEMPLATE.format(
        idx=idx,
        seq=",".join(seq),
        key=key,
        o1=step.options[0],
        o2=step.options[1],
        o3=step.options[2],
    )
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}, {"role": "user", "content": user}]
    t0 = time.perf_counter()
    out = ollama_chat_call(model, messages, options=STRICT_OPTIONS)
    dt = time.perf_counter() - t0
    choice = parse_choice(out or "")
    if verbose:
        print(f"\n--- Step {idx} ---\n{user}\nRAW: {out}")
    return choice, dt


# --- Logging helpers ---
RESULTS_CSV = "results.csv"
RESULTS_JSON = "results.json"


def log_result(entry: dict):
    # Append to CSV
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=entry.keys())
        if write_header:
            w.writeheader()
        w.writerow(entry)

    # Append to JSON array
    data = []
    if os.path.exists(RESULTS_JSON):
        try:
            with open(RESULTS_JSON) as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)


# --- Run episode ---
def run_episode(model: str, steps: int, seq: List[str], seed: int, verbose=False):
    plan = make_map(steps, seq, seed)
    correct = 0
    invalid = 0
    total_time = 0
    for s in plan:
        choice, dt = ask_model(model, s.idx, s, seq, verbose)
        total_time += dt
        if choice == s.correct:
            correct += 1
        elif choice is None:
            invalid += 1
    acc = 100.0 * correct / steps
    avg_ms = total_time / steps * 1000
    result = {
        "model": model,
        "steps": steps,
        "difficulty": len(seq),
        "accuracy": acc,
        "invalid": invalid,
        "avg_ms": avg_ms,
    }
    print(
        f"Model={model} | Steps={steps} | Difficulty={len(seq)} "
        f"| Accuracy={acc:.1f}% | Invalid={invalid} | Avg {avg_ms:.0f} ms"
    )
    log_result(result)
    return result


# --- Plotting ---
def update_plot(results: List[dict]):
    if not results:
        return
    plt.clf()
    difficulties = [r["difficulty"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    latencies = [r["avg_ms"] for r in results]

    plt.subplot(2, 1, 1)
    plt.plot(difficulties, accuracies, marker="o", label="Accuracy %")
    plt.ylabel("Accuracy %")
    plt.ylim(0, 105)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(difficulties, latencies, marker="o", color="orange", label="Latency (ms)")
    plt.ylabel("Latency (ms)")
    plt.xlabel("Difficulty")
    plt.legend()

    plt.pause(0.1)


# --- CLI ---
def main():
    p = argparse.ArgumentParser(
        description="AI Menu Navigator Benchmark with Live Graphs",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--model", default="gpt-oss", help="Model name to query (default: gpt-oss)")
    p.add_argument("--steps", type=int, default=10, help="Number of steps in the episode")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--difficulty", type=int, default=16, help="Sequence length / difficulty")
    p.add_argument("--verbose", action="store_true", help="Show prompts and raw outputs")
    p.add_argument(
        "--findmax",
        action="store_true",
        help="Increment difficulty until accuracy drops below 100%",
    )
    args = p.parse_args()

    plt.ion()  # interactive mode
    results = []

    if args.findmax:
        d = 1
        while True:
            seq = make_hex_sequence(d)
            res = run_episode(args.model, args.steps, seq, args.seed, args.verbose)
            results.append(res)
            update_plot(results)
            if res["accuracy"] < 100.0:
                print(f"\n>>> Max reliable difficulty = {d-1}")
                break
            d *= 2
    else:
        seq = make_hex_sequence(args.difficulty)
        res = run_episode(args.model, args.steps, seq, args.seed, args.verbose)
        results.append(res)
        update_plot(results)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
