# coder.py
"""
LangGraph + Cursor CLI iterative codegen POC (Colab-friendly)

What this does:
- Takes a USER prompt describing a desired POC
- Adds:
  (a) instruction to create unit tests
  (b) DOD/KPIs in quantitative terms
  (c) tests must check KPIs
- Calls Cursor CLI to generate/repair code + tests
- Runs pytest, captures feedback, and loops until KPIs pass (or max iterations)
- Exports final generated code to a *single* Jupyter-compatible .py file (with # %% cells)

Colab quickstart:
!pip -q install langgraph pytest
# Ensure Cursor CLI is installed and authenticated in the runtime.
# Optionally set CURSOR_CLI_CMD and CURSOR_CLI_ARGS env vars.

Then:
from coder import run_poc
path = run_poc(
  user_prompt="Build a POC that ...",
  kpis={
    "kpi_example": "Return JSON with keys ['a','b'] and values are ints; handle empty input gracefully"
  },
  output_ipy_path="/content/final_notebook.py",
  max_iters=8,
)
print("Wrote:", path)

Notes about Cursor CLI integration:
- Cursor CLI invocation differs by installation/version.
- This file uses a simple subprocess wrapper that sends the prompt via stdin and reads stdout.
- Customize with env vars:
    CURSOR_CLI_CMD  (default: "cursor")
    CURSOR_CLI_ARGS (default: "")
- If your Cursor CLI needs a specific subcommand, set CURSOR_CLI_ARGS accordingly.
"""

from __future__ import annotations

ALLOW_BLIND_EXECUTION = False

import dataclasses
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


CURSOR_TIMEOUT = int(os.environ.get("CURSOR_TIMEOUT_S", "3600"))

# ---------------------------
# Optional LangGraph import
# ---------------------------
_LANGGRAPH_AVAILABLE = True
try:
    from langgraph.graph import StateGraph, END
except Exception:
    _LANGGRAPH_AVAILABLE = False


# ---------------------------
# State definition
# ---------------------------
class RunMode(str, Enum):
    BLIND_AUTONOMOUS = "blind-autonomous"
    SAFE_INTERACTIVE = "safe-interactive"


@dataclass
class AgentState:
    user_prompt: str
    kpis: str              # Quantitative / formal criteria (human-readable)
    workspace_dir: str
    mode: "RunMode" = RunMode.SAFE_INTERACTIVE
    iteration: int = 0
    max_iters: int = 8
    model_notes: str = ""             # Optional user/system notes
    last_cursor_raw: str = ""
    last_parse_ok: bool = False
    manual_stop: bool = False

    # Paths
    solution_path: str = ""
    tests_path: str = ""

    # Test feedback
    pytest_returncode: Optional[int] = None
    pytest_stdout: str = ""
    pytest_stderr: str = ""
    failing_tests_estimate: Optional[int] = None

    # Convergence tracking
    history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    # Final export
    final_ipy_path: str = ""
    converged: bool = False
    stop_reason: str = ""


# ---------------------------
# Cursor CLI wrapper
# ---------------------------
def _cursor_cmd() -> List[str]:
    cmd = os.environ.get("CURSOR_CLI_CMD", "cursor").strip()
    args = os.environ.get("CURSOR_CLI_ARGS", "").strip()
    base = [cmd]
    if args:
        base += shlex.split(args)
    return base


def call_cursor_cli(prompt: str, timeout_s: int = CURSOR_TIMEOUT) -> str:
    """
    Calls Cursor CLI by sending `prompt` on stdin and returns stdout.
    You may need to configure CURSOR_CLI_CMD / CURSOR_CLI_ARGS for your environment.
    """
    from google.colab import userdata

    CURSOR_API_KEY = userdata.get('CURSOR_API_KEY')
    #cmd = _cursor_cmd()
    #cmd = f"cursor-agent --api-key {CURSOR_API_KEY} -p {prompt}"
    # Write prompt to a temporary file to avoid command line length issues
    prompt_filename = "prompt.txt"
    with open(prompt_filename, "w", encoding="utf-8") as f:
        f.write(prompt)


    with open(prompt_filename, "r", encoding="utf-8") as f:
        prompt = f.read()

    cmd = [
        "cursor-agent",
        "--api-key", CURSOR_API_KEY,
        "-p", prompt
    ]


    try:
        proc = subprocess.run(
            cmd,
            #input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Cursor CLI not found. Tried command: {cmd}. "
            f"Set CURSOR_CLI_CMD / CURSOR_CLI_ARGS or install Cursor CLI."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Cursor CLI timed out after {timeout_s}s. Command: {cmd}") from e

    # Prefer stdout; include stderr if stdout is empty for better debugging
    out = proc.stdout if proc.stdout.strip() else proc.stderr
    if not out.strip():
        out = f"(Cursor CLI produced no output)\nReturn code: {proc.returncode}\nSTDERR:\n{proc.stderr}"
    return out


# ---------------------------
# Prompt construction
# ---------------------------
def _format_kpis(kpis: Dict[str, str]) -> str:
    lines = []
    for i, (name, crit) in enumerate(kpis.items(), start=1):
        lines.append(f"{i}. {name}: {crit}")
    return "\n".join(lines) if lines else "(none provided)"


def get_pip_list():
    '''
    runs a shell script to get the list of installed pip packages and their versions
    '''
    result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()

def build_cursor_prompt(
    user_prompt: str,
    kpis,
    feedback: Optional[str] = None,
    iteration: int = 0,
) -> str:
    """
    Creates a single prompt for Cursor CLI that requests:
      - solution.py implementation
      - test_solution.py with pytest tests checking KPIs
    """
    kpi_block = kpis
    feedback_block = ""
    if feedback and feedback.strip():
        feedback_block = textwrap.dedent(f"""
        ## Feedback from last run (must fix)
        {feedback.strip()}
        """)

    # IMPORTANT: We force a strict output format to parse deterministically.
    return textwrap.dedent(f"""
    You are generating code for a Google Colab environment.

    ## USER request
    {user_prompt.strip()}

    ## Definition of Done (DOD) / KPIs (formal + testable)
    The solution is DONE only if *all* KPIs below are met and verified by automated unit tests:

    {kpi_block}

    ## Instructions
    1) Produce TWO files:
       - solution.py : the implementation
       - test_solution.py : pytest unit tests
    2) The tests MUST check that the KPIs are met (quantitatively and deterministically).
    3) The solution must be runnable in Colab with only standard pip installs (avoid obscure system deps).
    4) If assumptions are needed, encode them explicitly in code and in tests.
    5) Keep the API stable: tests should import from solution.py.
    6) Prefer small, reliable code, however it should fail on erros and not include try/catch and fallbacks, every error (from unresolved errors to runtime erros) should fail the tests. Include docstrings and type hints.
    7) there should be one function running the entire task asked in the user request (not the tests), that will not have any arguments, named main_notebook_call(), it will be tested as well by the tests to check that it doesn't crash.
    
    
    Note that the previous code generated might have errors, you need to fix them based on the feedback below.
    {feedback_block}
        
    ## Output format (STRICT)
    Output exactly two fenced code blocks, in this order:

    ```python file=solution.py
    <contents>
    ```

    ```python file=test_solution.py
    <contents>
    ```

    (No extra text outside those code blocks.)

    Keep in mind that these are the libraries currently installed in the environment and their versions:
    {get_pip_list()}

    """).strip()


# ---------------------------
# Output parsing + writing
# ---------------------------
_CODEBLOCK_RE = re.compile(
    r"```python\s+file=(?P<fname>[^\n\r]+)\s*\n(?P<code>.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def parse_cursor_output(raw: str) -> Dict[str, str]:
    """
    Returns {filename: code}. Expects solution.py and test_solution.py.
    """
    files: Dict[str, str] = {}
    for m in _CODEBLOCK_RE.finditer(raw):
        fname = m.group("fname").strip()
        code = m.group("code")
        files[fname] = code

    return files


def write_files(workspace: Path, files: Dict[str, str]) -> Tuple[Path, Path]:
    solution = workspace / "solution.py"
    tests = workspace / "test_solution.py"

    if "solution.py" not in files or "test_solution.py" not in files:
        raise ValueError(
            "Cursor output missing required files. "
            f"Found: {list(files.keys())[:10]}"
        )

    solution.write_text(files["solution.py"], encoding="utf-8")
    tests.write_text(files["test_solution.py"], encoding="utf-8")
    return solution, tests


def _archive_previous_versions(workspace: Path, iteration: int) -> None:
    archive_dir = workspace / "previous_versions"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("solution.py", "test_solution.py"):
        src = workspace / filename
        if not src.exists():
            continue
        archived = archive_dir / f"iter_{iteration}_{filename}"
        archived.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _archive_prompt_and_raw(workspace: Path, iteration: int, prompt: str, raw: str) -> None:
    archive_dir = workspace / "previous_versions"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / f"iter_{iteration}_prompt.txt").write_text(prompt, encoding="utf-8")
    (archive_dir / f"iter_{iteration}_cursor_raw.txt").write_text(raw, encoding="utf-8")


def _archive_test_results(
    workspace: Path, iteration: int, returncode: int, stdout: str, stderr: str
) -> None:
    archive_dir = workspace / "previous_versions"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / f"iter_{iteration}_pytest_returncode.txt").write_text(
        str(returncode), encoding="utf-8"
    )
    (archive_dir / f"iter_{iteration}_pytest_stdout.txt").write_text(stdout or "", encoding="utf-8")
    (archive_dir / f"iter_{iteration}_pytest_stderr.txt").write_text(stderr or "", encoding="utf-8")


# ---------------------------
# Pytest runner
# ---------------------------
_FAILCOUNT_RE = re.compile(r"=+\s+(\d+)\s+failed", re.IGNORECASE)


def run_pytest(workspace: Path, timeout_s: int = 1800) -> Tuple[int, str, str, Optional[int]]:
    """
    Runs pytest in workspace, returns (returncode, stdout, stderr, failing_tests_estimate).
    """
    cmd = [sys.executable, "-m", "pytest", "-q"]
    proc = subprocess.run(
        cmd,
        cwd=str(workspace),
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    out, err = proc.stdout, proc.stderr

    failing = None
    m = _FAILCOUNT_RE.search(out + "\n" + err)
    if m:
        try:
            failing = int(m.group(1))
        except Exception:
            failing = None

    return proc.returncode, out, err, failing


def _parse_mode(mode: str) -> RunMode:
    try:
        return RunMode(mode)
    except ValueError as exc:
        allowed = [m.value for m in RunMode]
        raise ValueError(f"Invalid mode: {mode}. Allowed: {allowed}") from exc


def _has_required_main_notebook_call_test(tests_path: Path) -> bool:
    if not tests_path.exists():
        return False
    content = tests_path.read_text(encoding="utf-8")
    # Require a pytest test that explicitly calls main_notebook_call().
    return bool(re.search(r"def\s+test_[\s\S]*?main_notebook_call\s*\(", content))


def _run_main_notebook_call(workspace: Path) -> Optional[str]:
    solution_path = workspace / "solution.py"
    if not solution_path.exists():
        return "solution.py not found for main_notebook_call() check."

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("solution", solution_path)
        if spec is None or spec.loader is None:
            return "Failed to import solution.py for main_notebook_call() check."
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        return f"Error importing solution.py for main_notebook_call(): {exc}"

    if not hasattr(module, "main_notebook_call"):
        return "solution.main_notebook_call() not found."

    try:
        module.main_notebook_call()
    except Exception as exc:
        return f"solution.main_notebook_call() raised an exception: {exc}"

    return None


# ---------------------------
# Feedback synthesis
# ---------------------------
def build_feedback_for_cursor(state: AgentState) -> str:
    """
    Construct feedback string passed back to Cursor.
    Keep it concise and actionable.
    """
    parts = []
    parts.append(f"Iteration: {state.iteration}/{state.max_iters}")

    #last solution.py and test_solution.py
    parts.append("Last generated solution.py:")
    if state.solution_path and Path(state.solution_path).exists():
        parts.append(Path(state.solution_path).read_text(encoding="utf-8").strip())
    else:
        parts.append("(not available)")
    parts.append("Last generated test_solution.py:")
    if state.tests_path and Path(state.tests_path).exists():
        parts.append(Path(state.tests_path).read_text(encoding="utf-8").strip())
    else:
        parts.append("(not available)")
    # Include KPI list again for focus
    parts.append("KPIs to satisfy:")
    parts.append(state.kpis)

    # Add pytest failure output
    if state.pytest_returncode is None:
        parts.append("Pytest did not run (no return code).")
    elif state.pytest_returncode == 0:
        parts.append("All tests passed (pytest return code 0).")
    else:
        parts.append("Pytest failed. Fix the issues below.\n")
        # Keep last ~2000 chars to avoid huge prompts
        combined = (state.pytest_stdout + "\n" + state.pytest_stderr).strip()
        if len(combined) > 2000:
            combined = combined[-2000:]
            combined = "(truncated tail)\n" + combined
        parts.append(combined)

    return "\n".join(parts).strip()


# ---------------------------
# Convergence heuristics
# ---------------------------
def estimate_distance_to_convergence(state: AgentState) -> Dict[str, Any]:
    """
    Heuristic signals to predict if we're far from convergence.
    These are *not* guarantees—just useful monitoring indicators.
    """
    hist = state.history
    signals: Dict[str, Any] = {}

    # Signal 1: failing tests trend
    fail_counts = [h.get("failing_tests_estimate") for h in hist if h.get("failing_tests_estimate") is not None]
    if len(fail_counts) >= 2:
        signals["failing_tests_trend"] = fail_counts[-1] - fail_counts[-2]
        signals["failing_tests_recent"] = fail_counts[-1]
    elif len(fail_counts) == 1:
        signals["failing_tests_recent"] = fail_counts[-1]

    # Signal 2: repeated identical failures (stuck)
    last_rcs = [h.get("pytest_returncode") for h in hist[-3:]]
    if len(last_rcs) == 3 and len(set(last_rcs)) == 1 and last_rcs[0] != 0:
        signals["stuck_returncode_3x"] = True
    else:
        signals["stuck_returncode_3x"] = False

    # Signal 3: parse failures
    parse_fails = [h.get("parse_ok") for h in hist[-3:]]
    signals["parse_instability"] = any(p is False for p in parse_fails) if parse_fails else False

    # Signal 4: runtime error class hints (from stdout/stderr)
    combined = (state.pytest_stdout + "\n" + state.pytest_stderr)
    if "ImportError" in combined or "ModuleNotFoundError" in combined:
        signals["likely_dependency_or_import_issue"] = True
    else:
        signals["likely_dependency_or_import_issue"] = False

    # Suggest interventions
    interventions: List[str] = []
    if signals.get("parse_instability"):
        interventions.append("Tighten output formatting instructions (STRICT fenced blocks only).")
    if signals.get("stuck_returncode_3x"):
        interventions.append("Escalate: ask Cursor to simplify design, rewrite from scratch, or reduce scope.")
    if signals.get("likely_dependency_or_import_issue"):
        interventions.append("Ask Cursor to remove non-standard deps or add pip-install guidance + skip unavailable deps.")
    if signals.get("failing_tests_trend") is not None and signals["failing_tests_trend"] >= 0:
        interventions.append("No improvement in failing tests: provide more targeted feedback snippets or pin an API.")
    signals["suggested_interventions"] = interventions

    return signals


# ---------------------------
# LangGraph nodes
# ---------------------------
def node_generate_with_cursor(state: AgentState) -> AgentState:
    workspace = Path(state.workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    feedback = None
    if state.iteration > 0:
        feedback = build_feedback_for_cursor(state)

    prompt = build_cursor_prompt(
        user_prompt=state.user_prompt,
        kpis=state.kpis,
        feedback=feedback,
        iteration=state.iteration,
    )

    print("Calling Cursor CLI...")
    raw = call_cursor_cli(prompt)
    _archive_prompt_and_raw(workspace, state.iteration, prompt, raw)
    state.last_cursor_raw = raw

    try:
        _archive_previous_versions(workspace, state.iteration)
        files = parse_cursor_output(raw)
        solution, tests = write_files(workspace, files)
        state.solution_path = str(solution)
        state.tests_path = str(tests)
        state.last_parse_ok = True
    except Exception as e:
        # Write raw output for debugging
        (workspace / "cursor_raw.txt").write_text(raw, encoding="utf-8")
        state.last_parse_ok = False
        state.pytest_returncode = 1
        state.pytest_stdout = ""
        state.pytest_stderr = f"Parse/write failure: {e}"

    return state


def node_run_tests(state: AgentState) -> AgentState:
    workspace = Path(state.workspace_dir)
    if not state.last_parse_ok:
        return state

    tests_path = Path(state.tests_path) if state.tests_path else (workspace / "test_solution.py")
    if not _has_required_main_notebook_call_test(tests_path):
        state.pytest_returncode = 1
        state.pytest_stdout = ""
        state.pytest_stderr = (
            "Missing required test: add a pytest test that calls main_notebook_call(). "
            "This iteration is failed until that test exists."
        )
        state.failing_tests_estimate = 1
        _archive_test_results(
            workspace,
            state.iteration,
            state.pytest_returncode,
            state.pytest_stdout,
            state.pytest_stderr,
        )
        state.history.append(
            {
                "iteration": state.iteration,
                "pytest_returncode": state.pytest_returncode,
                "failing_tests_estimate": state.failing_tests_estimate,
                "parse_ok": state.last_parse_ok,
                "ts": time.time(),
            }
        )
        return state

    if state.mode == RunMode.SAFE_INTERACTIVE:
        prompt = (
            "Safe-interactive mode: Review generated code before execution.\n"
            f"- solution: {state.solution_path}\n"
            f"- tests: {state.tests_path}\n"
            "Approve execution of tests? [y/N]: "
        )
        answer = input(prompt).strip().lower()
        if answer not in {"y", "yes"}:
            state.manual_stop = True
            state.stop_reason = "User declined execution in safe-interactive mode."
            return state

    print("Running pytest...")
    rc, out, err, failing = run_pytest(workspace)
    if rc == 0:
        print("Calling main_notebook_call() for explicit check...")
        main_call_error = _run_main_notebook_call(workspace)
        if main_call_error:
            rc = 1
            err = (err + "\n" if err else "") + f"main_notebook_call check failed: {main_call_error}"
            if not failing:
                failing = 1

    state.pytest_returncode = rc
    state.pytest_stdout = out
    state.pytest_stderr = err
    state.failing_tests_estimate = failing
    _archive_test_results(workspace, state.iteration, rc, out, err)

    # Record history snapshot for convergence monitoring
    state.history.append(
        {
            "iteration": state.iteration,
            "pytest_returncode": rc,
            "failing_tests_estimate": failing,
            "parse_ok": state.last_parse_ok,
            "ts": time.time(),
        }
    )

    return state


def node_check_done(state: AgentState) -> AgentState:
    if state.manual_stop:
        state.converged = False
        if not state.stop_reason:
            state.stop_reason = "Stopped by user."
    elif state.pytest_returncode == 0:
        state.converged = True
        state.stop_reason = "All tests passed (KPIs met)."
    elif state.iteration >= state.max_iters - 1:
        state.converged = False
        state.stop_reason = "Max iterations reached."
    else:
        state.converged = False
        state.stop_reason = "Continue."
    return state


def _should_continue(state: AgentState) -> str:
    if state.manual_stop:
        return "stop"
    return "stop" if state.converged or state.iteration >= state.max_iters - 1 else "continue"


def node_bump_iter(state: AgentState) -> AgentState:
    state.iteration += 1
    return state


# ---------------------------
# Export final to "ipython .py" (Jupyter/Colab compatible)
# ---------------------------
def export_to_ipy_py(solution_path: Path, out_path: Path) -> None:
    """
    Export solution.py into a single-file "notebook style" .py using cell markers.
    Colab can open this as a notebook-like script.
    """
    code = solution_path.read_text(encoding="utf-8").rstrip() + "\n"

    # Very simple cellization:
    # - One top cell with imports + module docstring
    # - One cell with the rest
    # This keeps it stable; you can expand later to parse sections.
    lines = code.splitlines()
    first_cell: List[str] = []
    rest: List[str] = []

    # Heuristic: put module docstring + imports into first cell until first non-import def/class
    seen_def = False
    for ln in lines:
        if re.match(r"^\s*(def|class)\s+\w+", ln):
            seen_def = True
        if not seen_def:
            first_cell.append(ln)
        else:
            rest.append(ln)

    content = []
    content.append("# %% [markdown]")
    content.append("# Generated POC (final) — exported by coder.py")
    content.append("# %%")
    content.extend(first_cell if first_cell else ["# (no header content)"])
    content.append("# %%")
    content.extend(rest if rest else ["# (no implementation content)"])

    out_path.write_text("\n".join(content).rstrip() + "\n", encoding="utf-8")


# ---------------------------
# Public API: run from Colab
# ---------------------------
def run_poc(
    user_prompt: str,
    kpis,
    output_ipy_path: str,
    workspace_dir: str = "./poc_workspace",
    max_iters: int = 8,
    model_notes: str = "",
    mode: RunMode = RunMode.SAFE_INTERACTIVE
) -> str:
    """
    Runs the LangGraph agent loop and writes a single final notebook-style .py file.

    Returns: output_ipy_path
    """


    if not _LANGGRAPH_AVAILABLE:
        raise RuntimeError(
            "langgraph is not installed. In Colab run: pip install langgraph\n"
            "If you want a non-LangGraph fallback, you can adapt this file easily."
        )

    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)

    state = AgentState(
        user_prompt=user_prompt,
        kpis=kpis,
        workspace_dir=str(ws),
        max_iters=max_iters,
        model_notes=model_notes,
        final_ipy_path=output_ipy_path,
        mode=mode,
    )

    # Build LangGraph
    graph = StateGraph(AgentState)
    graph.add_node("generate", node_generate_with_cursor)
    graph.add_node("test", node_run_tests)
    graph.add_node("check", node_check_done)
    graph.add_node("bump", node_bump_iter)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "test")
    graph.add_edge("test", "check")

    graph.add_conditional_edges(
        "check",
        _should_continue,
        {
            "continue": "bump",
            "stop": END,
        },
    )
    graph.add_edge("bump", "generate")

    app = graph.compile()

    final_state_as_dict = app.invoke(state)
    final_state: AgentState = dataclasses.replace(state, **final_state_as_dict)

    # Write convergence diagnostics for the user to inspect
    signals = estimate_distance_to_convergence(final_state)
    (ws / "convergence_signals.json").write_text(json.dumps(signals, indent=2), encoding="utf-8")
    (ws / "history.json").write_text(json.dumps(final_state.history, indent=2), encoding="utf-8")
    (ws / "last_cursor_raw.txt").write_text(final_state.last_cursor_raw or "", encoding="utf-8")
    (ws / "last_pytest_stdout.txt").write_text(final_state.pytest_stdout or "", encoding="utf-8")
    (ws / "last_pytest_stderr.txt").write_text(final_state.pytest_stderr or "", encoding="utf-8")

    # Export final code (even if not converged, export the latest solution so user can inspect)
    sol_path = Path(final_state.solution_path) if final_state.solution_path else (ws / "solution.py")
    out_path = Path(output_ipy_path)
    export_to_ipy_py(sol_path, out_path)

    # Also store a small summary
    summary = {
        "converged": final_state.converged,
        "stop_reason": final_state.stop_reason,
        "iterations": final_state.iteration + 1,  # since iteration is 0-based
        "workspace": str(ws.resolve()),
        "exported_ipy_py": str(out_path.resolve()),
        "signals": signals,
    }
    (ws / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return output_ipy_path


# ---------------------------
# Convergence strategy suggestions (for README/logging)
# ---------------------------
def convergence_playbook() -> Dict[str, List[str]]:
    """
    Suggested methods to predict if you're far from convergence and what to do.
    Returns a structured playbook you can print/log in Colab.
    """
    return {
        "Predict_far_from_convergence": [
            "Failing tests not decreasing over 2–3 iterations (flat or increasing fail count).",
            "Repeated same exception class (e.g., ImportError) across iterations.",
            "Large, oscillating diffs: code changes drastically each iteration but failures persist.",
            "Frequent format/parse errors (model not following output contract).",
            "Test runtime or flakiness increases (non-deterministic behavior).",
        ],
        "Mitigations": [
            "Tighten the output contract (STRICT fenced blocks, no extra prose).",
            "Add a 'rewrite from scratch' instruction after N stuck iterations.",
            "Reduce scope: split KPIs into stages; gate on a subset first, then expand.",
            "Pin a stable public API in solution.py (functions/classes) and forbid renaming.",
            "Add 'minimal dependencies' rule; ask to remove optional deps if missing in Colab.",
            "Introduce a 'critic' step: have Cursor list root causes and a patch plan before coding.",
            "Use a budget: if no progress, stop early and surface best-effort artifact + diagnostics.",
        ],
    }


def run_coder(user_prompt, mode="safe-interactive"):
    '''
    example user_prompt = "Create a tiny POC with an add() function."
    '''
    # Simple local smoke usage (requires Cursor CLI + langgraph + pytest).
    generate_kpis_with_cursor = True
    mode = _parse_mode(mode)

    if not ALLOW_BLIND_EXECUTION and mode == RunMode.BLIND_AUTONOMOUS:
        raise ValueError("Blind execution is not allowed. Please explictly change code to ALLOW_BLIND_EXECUTION=True if you read the README.md file and understand the risks or use safe-interactive mode.")

    kpi_prompt = textwrap.dedent(f"""
    You are an expert software engineer.

    Given the following user prompt - to create a colab script, generate a list of kpis (as free text) that can be used to verify the solution meets the user's requirements.
    It is important that every solution will not include try/catch and fallbacks and that every error (from unresolved errors to runtime erros) should fail the tests - 
    make sure the KPIs include this requirements and that it is is tested as well (tests that assess the code itself))
    """).strip()
    kpis = call_cursor_cli(kpi_prompt)
        
    out = run_poc(
        user_prompt=user_prompt,
        kpis=kpis,
        output_ipy_path="./final_notebook.py",
        max_iters=10,
        mode=mode,
    )
    print("Exported:", out)
  
    #print the code that will be run:
    print(
        Path(out).read_text(encoding="utf-8")
    )
    #run the notbebok with ipython:


    os.system(f"ipython {out}")
