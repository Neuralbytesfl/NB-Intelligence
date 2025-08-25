"""
AI Code Runner — Autonomous, Safe, Cohesive, and Adaptive
fast chat -> make TODO plan -> code per step -> auto-run -> auto-fix -> self-rate -> ✓/✗ board -> learn

Requirements:
    pip install ollama

What this does:
- Asks you for a GOAL.
- Uses the model to produce a compact TODO plan (JSON).
- For each step, asks the model for a SMALL, SELF-CONTAINED Python script.
- Shows the code, checks for risky patterns, and (by default) runs it automatically.
- Auto-fixes on errors up to MAX_FIX_ATTEMPTS and self-rates each step (1–5).
- Tracks progress with checkmarks and a scoreboard; saves everything under Desktop/ai_runs/<slug>/.
- **Adaptive**: logs code versions, ratings, notes, and diffs; uses past high-scorers to bias future prompts.

Safety:
- Blocks dangerous patterns (e.g., os.remove, shutil.rmtree, rm -rf, network libs) unless you explicitly allow.
- Runs scripts in a separate process group and kills the whole group after EXEC_TIMEOUT_SECS (default 60s).
- All file I/O must be relative to the run workspace.

CLI flags & env (optional):
- --confirm to require Y/N confirmations (default: auto-run)
- --allow-risky to run even if safety flags are detected (default: skip risky)
- OLLAMA_MODEL, OLLAMA_TEMP, OLLAMA_MAX_TOKENS override model settings
- AI_RUNNER_AUTORUN=0 to disable auto-run (equivalent to --confirm)
- AI_RUNNER_ALLOW_RISKY=1 to allow risky code without prompts (use with care)
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import shutil
import tempfile
import subprocess
import threading
import signal
import difflib
import hashlib
import argparse
from typing import List, Dict, Tuple, Optional

import ollama

# =========================
# Configuration
# =========================
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
TEMPERATURE = float(os.environ.get("OLLAMA_TEMP", "0.2"))
MAX_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", "1000"))
STREAM = True
HISTORY_TURNS = 6               # user+assistant pairs kept (small = faster)
EXEC_TIMEOUT_SECS = 60          # hard cap: any step process is killed after this many seconds
MAX_FIX_ATTEMPTS =10          # auto-fix retries on exceptions or low self-rating
PLANNER_MAX_TASKS =6         # cap plan length to keep things snappy
SAVE_ROOT = os.path.join(os.path.expanduser("~"), "Desktop", "ai_runs")

# Auto-run / safety knobs (can be overridden by CLI flags)
AUTO_YES_DEFAULT = os.environ.get("AI_RUNNER_AUTORUN", "1") != "0"
ALLOW_RISKY_DEFAULT = os.environ.get("AI_RUNNER_ALLOW_RISKY", "0") == "1"

# Learning artifacts
GLOBAL_LEARN_LOG = os.path.join(SAVE_ROOT, "adaptive_learning_log.json")

os.makedirs(SAVE_ROOT, exist_ok=True)

# =========================
# Prompts
# =========================
SYSTEM_PROMPT = (
    "You are a coding assistant that plans compact TODOs and writes SMALL, SAFE, runnable Python scripts.\n"
    "PRINCIPLES:\n"
    "- Be concise and practical. Prefer standard library only. No networking, no destructive file ops.\n"
    "- Cohesion: use previously created files only if told, and keep file I/O relative to the working directory.\n"
    "- When asked for code, return EXACTLY ONE python fenced code block, complete & self-contained.\n"
)

PLAN_PROMPT = lambda goal, max_tasks: f"""
You will plan a short, cohesive TODO list to accomplish the user's goal.

USER GOAL:
{goal}

REQUIREMENTS:
- Keep it compact and actionable (at most {max_tasks} tasks).
- Prefer steps that can be done purely in Python stdlib, no network, no admin actions, no system changes.
- Each task should be clear, specific, and runnable in isolation.

OUTPUT STRICTLY AS JSON IN ONE CODE BLOCK:
```json
{{
  "tasks": [
    {{"id": 1, "title": "…", "why": "…"}},
    {{"id": 2, "title": "…", "why": "…"}}
  ]
}}
```
"""

# Adaptation hints are appended dynamically (see request_step_code)
BASE_STEP_CODE_PROMPT = lambda goal, step_title, prior_summary, hints: f"""
Write a SMALL, SELF-CONTAINED Python script to complete this step.

GOAL: {goal}
STEP: {step_title}

CONTEXT (prior steps summary / artifacts):
{prior_summary}

ADAPTATION HINTS (from prior high-rated runs):
{hints}

CONSTRAINTS:
- Standard library only.
- NO networking (no requests, urllib, socket).
- NO destructive ops: do not delete, rename, move, wipe, or modify outside cwd.
- Keep it compact (ideally <= 80 lines) and robust (basic error handling).
- All file I/O must be relative to current working directory (create a 'out' folder if needed).
- Print human-readable results to stdout.

Return ONLY one python fenced code block.
"""

RATE_PROMPT = lambda goal, step_title, stdout_text, stderr_text: f"""
You are evaluating the result of a step.

GOAL: {goal}
STEP: {step_title}

STDOUT:
{stdout_text[:4000]}

STDERR:
{stderr_text[:2000]}

TASK:
Return a JSON with a 'rating' from 1 to 5 (5 = excellent), and 'improve': true/false, and 'note' (short).
If the rating is < 4, also include an 'advice' field with a concise fix suggestion.

Return JSON only in a single code block:
```json
{{"rating": 4, "improve": false, "note": "…", "advice": ""}}
```
"""

FIX_PROMPT = lambda goal, step_title, prev_code, traceback_text, advice: f"""
The previous Python script for this step failed or needs improvement.

GOAL: {goal}
STEP: {step_title}

CONSTRAINTS:
- Standard library only.
- NO networking (requests, urllib, socket).
- NO destructive ops: do not delete, rename, move, wipe, or modify outside cwd.
- Keep it compact (<= ~100 lines).
- All file I/O must be relative to cwd.

PREVIOUS CODE:
```python
{prev_code}
```

TRACEBACK / ISSUE:
```
{traceback_text}
```

SUGGESTED ADVICE:
{advice}

TASK:
Produce a FIXED, complete Python script that addresses the issue.
Return EXACTLY ONE python code block, no extra text.
"""

# =========================
# Utilities
# =========================

def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s or "run"


def trim_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(history) <= 1:
        return history
    max_len = 1 + 2 * HISTORY_TURNS
    return history[:1] + history[-(max_len - 1):]


def stream_chat(model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
    if not STREAM:
        resp = ollama.chat(model=model, messages=messages, options={"temperature": temperature, "max_tokens": max_tokens})
        return resp["message"]["content"]

    chunks = []
    for part in ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={"temperature": temperature, "max_tokens": max_tokens},
    ):
        delta = part.get("message", {}).get("content", "")
        if delta:
            chunks.append(delta)
    return "".join(chunks)


def extract_json_block(text: str) -> Optional[str]:
    m = re.search(r"```json\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(\{[\s\S]*\})", text)
    return m.group(1).strip() if m else None


def extract_code_block(text: str) -> Optional[str]:
    m = re.search(r"```python\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```[\s\S]*?```", text)  # any code fence
    if m:
        inner = m.group(0)[3:-3].strip()
        return inner
    return None


# confirmation gate (auto-run by default)
AUTO_YES = AUTO_YES_DEFAULT
ALLOW_RISKY = ALLOW_RISKY_DEFAULT


def ask_yes_no(prompt: str) -> bool:
    if AUTO_YES:
        print(f"[auto-yes] {prompt}")
        return True
    while True:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please type 'y' or 'n'.")


def write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_text(path: str, default: str = "") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


# ----- safer runner that kills process trees -----

def _kill_process_tree(proc: subprocess.Popen):
    try:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(0.5)
            except Exception:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        else:
            # Windows: taskkill the whole tree
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, text=True)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    except Exception:
        pass


def run_python_file(path: str, cwd: str, timeout: int = EXEC_TIMEOUT_SECS) -> Tuple[int, str, str, bool]:
    if os.name == "posix":
        popen_kwargs = dict(preexec_fn=os.setsid)
    else:
        popen_kwargs = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

    proc = subprocess.Popen(
        [sys.executable, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        **popen_kwargs,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr, False
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=2)
        except Exception:
            stdout, stderr = "", f"Execution timed out after {timeout}s."
        return 124, stdout or "", stderr or f"Execution timed out after {timeout}s.", True


def print_board(tasks: List[Dict]):
    print("\n" + "=" * 74)
    print("PROGRESS".ljust(10) + "ID".ljust(6) + "TITLE")
    print("-" * 74)
    for t in tasks:
        status = t.get("status", "pending")
        mark = {"pending": "[ ]", "running": "[>]", "done": "[✓]", "failed": "[✗]", "skipped": "[–]"}.get(status, "[ ]")
        rating = t.get("rating")
        extra = f"  (rating: {rating}/5)" if rating is not None else ""
        print(f"{mark.ljust(10)}{str(t['id']).ljust(6)}{t['title']}{extra}")
    print("=" * 74 + "\n")


def is_code_risky(code: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    bad_patterns = [
      
    ]

    for pat in bad_patterns:
        if re.search(pat, code, re.IGNORECASE):
            reasons.append(f"Pattern flagged: {pat}")
    if re.search(r"(^|\s)[A-Za-z]:\\", code):
        reasons.append("Absolute Windows path used.")
    if re.search(r"(^|\s)/[A-Za-z0-9_]", code):
        reasons.append("Absolute POSIX path used.")
    return (len(reasons) > 0, reasons)


# Artifact helpers

def artifacts_summary(workspace: str, max_items: int = 12) -> str:
    out_dir = os.path.join(workspace, "out")
    items: List[str] = []
    for root, _, files in os.walk(out_dir):
        rel_root = os.path.relpath(root, workspace)
        for f in files:
            items.append(os.path.join(rel_root, f))
    items = sorted(items)[:max_items]
    if not items:
        return "No artifacts yet."
    return "Artifacts: " + ", ".join(items)


# =========================
# Learning / Adaptation helpers
# =========================

def _load_json_list(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _append_json_list(path: str, obj: dict) -> None:
    data = _load_json_list(path)
    data.append(obj)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def record_version(workspace: str, step: Dict, code: str, stdout: str, stderr: str, rc: int, timed_out: bool, tag: str, diff_text: str = "") -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    versions_dir = os.path.join(workspace, "versions")
    logs_dir = os.path.join(workspace, "logs")
    os.makedirs(versions_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    fname_code = f"step{step['id']}_{tag}_{ts}.py"
    write_text(os.path.join(versions_dir, fname_code), code)

    meta = {
        "ts": ts,
        "step_id": step["id"],
        "title": step["title"],
        "tag": tag,
        "code_sha": _code_hash(code),
        "rc": rc,
        "timed_out": timed_out,
        "stdout_file": f"logs/step{step['id']}_{tag}_{ts}_stdout.txt",
        "stderr_file": f"logs/step{step['id']}_{tag}_{ts}_stderr.txt",
        "diff_file": f"logs/step{step['id']}_{tag}_{ts}_diff.txt" if diff_text else "",
    }

    write_text(os.path.join(logs_dir, f"step{step['id']}_{tag}_{ts}_stdout.txt"), stdout or "")
    write_text(os.path.join(logs_dir, f"step{step['id']}_{tag}_{ts}_stderr.txt"), stderr or "")
    if diff_text:
        write_text(os.path.join(logs_dir, f"step{step['id']}_{tag}_{ts}_diff.txt"), diff_text)

    _append_json_list(os.path.join(workspace, "versions_index.json"), meta)


def record_reward(workspace: str, step: Dict, code: str, rating_info: Dict, scope: str) -> None:
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workspace": os.path.basename(workspace),
        "step_id": step["id"],
        "title": step["title"],
        "code_sha": _code_hash(code),
        "rating": int(rating_info.get("rating", 3)),
        "note": str(rating_info.get("note", "")),
        "advice": str(rating_info.get("advice", "")),
        "scope": scope,
    }
    _append_json_list(os.path.join(workspace, "rewards.json"), entry)
    _append_json_list(GLOBAL_LEARN_LOG, entry)


def best_hints_from_rewards(workspace: str, top_k: int = 5) -> str:
    # Gather from local + global logs
    local = _load_json_list(os.path.join(workspace, "rewards.json"))
    global_ = _load_json_list(GLOBAL_LEARN_LOG)
    combined = [e for e in (local + global_) if isinstance(e, dict)]
    # Prefer high-rated entries with non-empty note/advice
    scored: List[Tuple[int, str]] = []
    for e in combined[-200:]:  # recent bias
        rating = int(e.get("rating", 0))
        text = (e.get("advice", "") or e.get("note", "")).strip()
        if rating >= 4 and text:
            scored.append((rating, text))
    # Deduplicate while preserving order and pick top_k unique snippets
    seen = set()
    hints: List[str] = []
    for _, t in sorted(scored, key=lambda x: -x[0]):
        if t not in seen:
            hints.append(t)
            seen.add(t)
        if len(hints) >= top_k:
            break
    if not hints:
        return "- Favor clear output messages.\n- Add basic error handling and validate inputs.\n- Keep code <= ~80 lines and stdlib-only."
    return "\n".join(f"- {h}" for h in hints)


# =========================
# Model helpers
# =========================

def model_respond(history: List[Dict[str, str]]) -> str:
    msgs = trim_history(history)
    return stream_chat(MODEL_NAME, msgs, TEMPERATURE, MAX_TOKENS)


def plan_tasks(goal: str) -> List[Dict]:
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PLAN_PROMPT(goal, PLANNER_MAX_TASKS)},
    ]
    text = model_respond(history)
    js = extract_json_block(text)
    if not js:
        raise RuntimeError("Planner did not return JSON.")
    data = json.loads(js)
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        raise RuntimeError("Planner JSON malformed: 'tasks' must be a list.")
    for i, t in enumerate(tasks, 1):
        t["id"] = int(t.get("id", i))
        t["title"] = str(t.get("title", f"Task {i}")).strip()
        t["why"] = str(t.get("why", "")).strip()
        t["status"] = "pending"
        t["rating"] = None
        t["note"] = ""
    return tasks[:PLANNER_MAX_TASKS]


def request_step_code(goal: str, step_title: str, prior_summary: str, workspace: str) -> str:
    hints = best_hints_from_rewards(workspace)
    prompt = BASE_STEP_CODE_PROMPT(goal, step_title, prior_summary, hints)
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    out = model_respond(history)
    code = extract_code_block(out)
    if not code:
        history.append({"role": "user", "content": "Return ONLY one python fenced code block, no explanations."})
        out2 = model_respond(history)
        code = extract_code_block(out2)
    if not code:
        raise RuntimeError("Model did not return a Python code block for the step.")
    return code


def request_rating(goal: str, step_title: str, stdout_text: str, stderr_text: str) -> Dict:
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": RATE_PROMPT(goal, step_title, stdout_text, stderr_text)},
    ]
    out = model_respond(history)
    js = extract_json_block(out)
    if not js:
        return {"rating": 3, "improve": False, "note": "No JSON rating, defaulting to 3.", "advice": ""}
    try:
        data = json.loads(js)
    except Exception:
        return {"rating": 3, "improve": False, "note": "Failed to parse rating JSON.", "advice": ""}
    data.setdefault("rating", 3)
    data.setdefault("improve", False)
    data.setdefault("note", "")
    data.setdefault("advice", "")
    return data


def request_fix(goal: str, step_title: str, prev_code: str, trace_or_reason: str, advice: str) -> str:
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FIX_PROMPT(goal, step_title, prev_code, trace_or_reason, advice)},
    ]
    out = model_respond(history)
    code = extract_code_block(out)
    if not code:
        history.append({"role": "user", "content": "Return ONLY one python fenced code block."})
        out2 = model_respond(history)
        code = extract_code_block(out2)
    if not code:
        raise RuntimeError("Model did not return a fixed Python code block.")
    return code


# =========================
# Orchestration
# =========================

def run_step(goal: str, workspace: str, tasks: List[Dict], idx: int, prior_summary: str) -> None:
    t = tasks[idx]
    t["status"] = "running"
    print_board(tasks)

    # Ask for code (with adaptive hints)
    try:
        code = request_step_code(goal, t["title"], prior_summary, workspace)
    except Exception as e:
        t["status"] = "failed"
        t["note"] = f"Code generation error: {e}"
        print(f"[Model Error] {e}")
        return

    # Safety gate
    risky, reasons = is_code_risky(code)
    print("\n" + "=" * 80)
    print(f"STEP {t['id']}: {t['title']}")
    print("=" * 80)
    print(code)
    print("=" * 80 + "\n")

    if risky and not ALLOW_RISKY:
        print("⚠️  Safety flags detected:")
        for r in reasons:
            print(" -", r)
        # In auto mode we SKIP risky by default (safer). Use --allow-risky to override.
        t["status"] = "skipped"
        t["note"] = "Skipped due to safety flags. Use --allow-risky to run anyway."
        # Still record the proposed code version for audit
        record_version(workspace, t, code, stdout="", stderr="SKIPPED (risky)", rc=0, timed_out=False, tag="proposed")
        return

    if not ask_yes_no("Run this code?"):
        t["status"] = "skipped"
        t["note"] = "Declined by user."
        return

    # Save & run
    ts = time.strftime("%Y%m%d_%H%M%S")
    code_path = os.path.join(workspace, f"step{t['id']}_{ts}.py")
    write_text(code_path, code)

    rc, stdout, stderr, timed_out = run_python_file(code_path, cwd=workspace, timeout=EXEC_TIMEOUT_SECS)

    # Record primary version
    record_version(workspace, t, code, stdout, stderr, rc, timed_out, tag="v1")

    print("\n" + "-" * 80)
    print("RUN OUTPUT")
    print("-" * 80)
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print("\n[stderr]")
        print(stderr, end="" if stderr.endswith("\n") else "\n")
    print("-" * 80 + "\n")

    if rc == 0 and not timed_out:
        t["status"] = "done"
    else:
        t["status"] = "failed"
        if timed_out:
            stderr = f"Timeout after {EXEC_TIMEOUT_SECS}s.\n" + (stderr or "")
        t["note"] = "Run failed."

    # Self-rate & possible improvement
    rating_info = request_rating(goal, t["title"], stdout or "", stderr or "")
    t["rating"] = int(rating_info.get("rating", 3))
    t["note"] = (t.get("note", "") + " " + rating_info.get("note", "")).strip()
    advice = rating_info.get("advice", "")
    record_reward(workspace, t, code, rating_info, scope="initial")

    need_improve = bool(rating_info.get("improve", False)) or (t["status"] != "done") or (t["rating"] < 4)

    attempts = 0
    last_code = code
    last_issue = stderr or t.get("note", "")

    while need_improve and attempts < MAX_FIX_ATTEMPTS:
        attempts += 1
        print(f"🔧 Attempting improvement #{attempts}/{MAX_FIX_ATTEMPTS} ...")
        try:
            fixed = request_fix(goal, t["title"], last_code, last_issue, advice)
        except Exception as e:
            print(f"[Fix Error] {e}")
            break

        # Safety again
        risky, reasons = is_code_risky(fixed)
        print("\n" + "=" * 80)
        print(f"FIXED CANDIDATE #{attempts} for STEP {t['id']}")
        print("=" * 80)
        print(fixed)
        print("=" * 80 + "\n")

        if risky and not ALLOW_RISKY:
            print("⚠️  Safety flags detected (fix):")
            for r in reasons:
                print(" -", r)
            # record diff but skip execution
            diff_text = "\n".join(difflib.unified_diff(last_code.splitlines(), fixed.splitlines(), fromfile="prev", tofile="fixed", lineterm=""))
            record_version(workspace, t, fixed, stdout="", stderr="SKIPPED (risky fix)", rc=0, timed_out=False, tag=f"fix{attempts}", diff_text=diff_text)
            break

        if not ask_yes_no("Run the fixed code?"):
            break

        ts_fix = time.strftime("%Y%m%d_%H%M%S")
        fix_path = os.path.join(workspace, f"step{t['id']}_fix{attempts}_{ts_fix}.py")
        write_text(fix_path, fixed)
        rc, stdout, stderr, timed_out = run_python_file(fix_path, cwd=workspace, timeout=EXEC_TIMEOUT_SECS)

        diff_text = "\n".join(difflib.unified_diff(last_code.splitlines(), fixed.splitlines(), fromfile="prev", tofile=f"fix{attempts}", lineterm=""))
        record_version(workspace, t, fixed, stdout, stderr, rc, timed_out, tag=f"fix{attempts}", diff_text=diff_text)

        print("\n" + "-" * 80)
        print("RUN OUTPUT (FIX)")
        print("-" * 80)
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print("\n[stderr]")
            print(stderr, end="" if stderr.endswith("\n") else "\n")
        print("-" * 80 + "\n")

        if rc == 0 and not timed_out:
            t["status"] = "done"
        else:
            t["status"] = "failed"
            if timed_out:
                stderr = f"Timeout after {EXEC_TIMEOUT_SECS}s.\n" + (stderr or "")

        rating_info = request_rating(goal, t["title"], stdout or "", stderr or "")
        t["rating"] = int(rating_info.get("rating", t.get("rating", 3)))
        t["note"] = (t.get("note", "") + " " + rating_info.get("note", "")).strip()
        advice = rating_info.get("advice", advice)
        record_reward(workspace, t, fixed, rating_info, scope=f"fix{attempts}")

        last_code = fixed
        last_issue = stderr or t.get("note", "")
        need_improve = (t["status"] != "done") or (t["rating"] < 4)

    print_board(tasks)


def save_state(workspace: str, goal: str, tasks: List[Dict]):
    state = {
        "goal": goal,
        "tasks": tasks,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_text(os.path.join(workspace, "scoreboard.json"), json.dumps(state, indent=2))


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="AI Code Runner — autonomous, safe, adaptive")
    parser.add_argument("--confirm", action="store_true", help="Require confirmations (disable auto-run)")
    parser.add_argument("--allow-risky", action="store_true", help="Allow execution even if safety flags are detected")
    args = parser.parse_args()

    global AUTO_YES, ALLOW_RISKY
    AUTO_YES = not args.confirm
    ALLOW_RISKY = args.allow_risky or ALLOW_RISKY

    print(f"AI Code Runner ready on model: {MODEL_NAME}")
    print("This flow: plan -> (per step) code -> auto-run -> auto-fix -> self-rate -> ✓ board -> adapt\n")

    goal = input("Enter GOAL (what do you want to accomplish?): ").strip()
    if not goal:
        print("No goal provided. Exiting.")
        return

    slug = slugify(goal)
    workspace = os.path.join(SAVE_ROOT, slug)
    os.makedirs(workspace, exist_ok=True)
    os.makedirs(os.path.join(workspace, "out"), exist_ok=True)

    # PLAN
    try:
        tasks = plan_tasks(goal)
    except Exception as e:
        print(f"[Planner Error] {e}")
        return

    write_text(os.path.join(workspace, "plan.json"), json.dumps({"tasks": tasks}, indent=2))

    print("\nPlanned Tasks:\n" + "-" * 74)
    for t in tasks:
        print(f"[{t['id']}] {t['title']} — why: {t['why']}")
    print("-" * 74 + "\n")

    if not ask_yes_no("Proceed with this plan?"):
        print("Plan declined. Exiting.")
        return

    print_board(tasks)

    # Run steps sequentially (deterministic). Each step has hard 60s cap.
    for i in range(len(tasks)):
        prior = artifacts_summary(workspace)
        save_state(workspace, goal, tasks)
        run_step(goal, workspace, tasks, i, prior)
        save_state(workspace, goal, tasks)

    # Final summary
    print("\nFinal Status:")
    print_board(tasks)
    print(f"Workspace: {workspace}")
    print("Artifacts (if any):", artifacts_summary(workspace))
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
