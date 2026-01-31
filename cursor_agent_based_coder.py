"""
cursor_agent_based_coder.py

Config-driven Cursor CLI loop for Colab/IPython.

This version avoids a hardcoded automata/graph and instead uses prompt
templates defined in markdown files referenced by a JSON config.

Quickstart (Colab):
!pip -q install pytest
!pip -q install cursor-agent  # or ensure cursor-agent is available

from cursor_agent_based_coder import run_poc_with_config
path = run_poc_with_config(
    config_path="./cursor_agent_config/config.json",
    user_prompt="Build a POC that ...",
    kpis=None,  # allow config to generate KPIs
    output_ipy_path="/content/final_notebook.py",
    max_iters=8,
)
print("Wrote:", path)
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CURSOR_TIMEOUT_DEFAULT = int(os.environ.get("CURSOR_TIMEOUT_S", "3600"))


@dataclass
class AgentConfig:
    cursor_cli_cmd: str
    cursor_cli_args: List[str]
    cursor_timeout_s: int
    workspace_dir: str
    max_iters: int
    output_ipy_path: str
    prompts: Dict[str, str]
    output_files: List[str]
    pytest_timeout_s: int


@dataclass
class RunState:
    iteration: int = 0
    converged: bool = False
    stop_reason: str = ""
    last_cursor_raw: str = ""
    last_parse_ok: bool = False
    pytest_returncode: Optional[int] = None
    pytest_stdout: str = ""
    pytest_stderr: str = ""
    failing_tests_estimate: Optional[int] = None
    history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    solution_path: str = ""
    tests_path: str = ""


def _load_config(config_path: str) -> AgentConfig:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return AgentConfig(
        cursor_cli_cmd=config.get("cursor_cli_cmd", "cursor-agent"),
        cursor_cli_args=config.get("cursor_cli_args", []),
        cursor_timeout_s=int(config.get("cursor_timeout_s", CURSOR_TIMEOUT_DEFAULT)),
        workspace_dir=config.get("workspace_dir", "./poc_workspace"),
        max_iters=int(config.get("max_iters", 8)),
        output_ipy_path=config.get("output_ipy_path", "./final_notebook.py"),
        prompts=config.get("prompts", {}),
        output_files=config.get("output_files", ["solution.py", "test_solution.py"]),
        pytest_timeout_s=int(config.get("pytest_timeout_s", 1800)),
    )


def _get_cursor_api_key() -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get("CURSOR_API_KEY")
    except Exception:
        return os.environ.get("CURSOR_API_KEY")


def _cursor_cmd(config: AgentConfig) -> List[str]:
    base = [config.cursor_cli_cmd]
    if config.cursor_cli_args:
        base.extend(config.cursor_cli_args)
    return base


def call_cursor_cli(prompt: str, config: AgentConfig) -> str:
    """
    Calls Cursor CLI by sending `prompt` on stdin and returns stdout.
    """
    api_key = _get_cursor_api_key()
    cmd = _cursor_cmd(config)
    if api_key and "cursor-agent" in cmd[0]:
        cmd = cmd + ["--api-key", api_key]

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=config.cursor_timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Cursor CLI not found. Tried command: {cmd}. "
            "Install cursor-agent or set cursor_cli_cmd in config."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Cursor CLI timed out after {config.cursor_timeout_s}s. Command: {cmd}"
        ) from exc

    out = proc.stdout if proc.stdout.strip() else proc.stderr
    if not out.strip():
        out = (
            "(Cursor CLI produced no output)\n"
            f"Return code: {proc.returncode}\nSTDERR:\n{proc.stderr}"
        )
    return out


def _read_prompt_template(prompt_path: str) -> str:
    return Path(prompt_path).read_text(encoding="utf-8")


def _format_prompt(template: str, data: Dict[str, str]) -> str:
    try:
        return template.format(**data)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(
            f"Prompt template missing variable: {missing}. "
            "Check the markdown template placeholders."
        ) from exc


def get_pip_list() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.stdout.strip()


_CODEBLOCK_RE = re.compile(
    r"```python\s+file=(?P<fname>[^\n\r]+)\s*\n(?P<code>.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def parse_cursor_output(raw: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for m in _CODEBLOCK_RE.finditer(raw):
        fname = m.group("fname").strip()
        code = m.group("code")
        files[fname] = code
    return files


def write_files(workspace: Path, files: Dict[str, str], required: List[str]) -> Tuple[Path, Path]:
    for req in required:
        if req not in files:
            raise ValueError(
                "Cursor output missing required files. "
                f"Expected: {required}. Found: {list(files.keys())[:10]}"
            )
    solution = workspace / required[0]
    tests = workspace / required[1]
    solution.write_text(files[required[0]], encoding="utf-8")
    tests.write_text(files[required[1]], encoding="utf-8")
    return solution, tests


def _archive_previous_versions(workspace: Path, iteration: int, required: List[str]) -> None:
    archive_dir = workspace / "previous_versions"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for filename in required:
        src = workspace / filename
        if not src.exists():
            continue
        archived = archive_dir / f"iter_{iteration}_{filename}"
        archived.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


_FAILCOUNT_RE = re.compile(r"=+\s+(\d+)\s+failed", re.IGNORECASE)


def run_pytest(workspace: Path, timeout_s: int) -> Tuple[int, str, str, Optional[int]]:
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


def export_to_ipy_py(solution_path: Path, out_path: Path) -> None:
    code = solution_path.read_text(encoding="utf-8").rstrip() + "\n"
    lines = code.splitlines()
    first_cell: List[str] = []
    rest: List[str] = []
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
    content.append("# Generated POC (final) — exported by cursor_agent_based_coder.py")
    content.append("# %%")
    content.extend(first_cell if first_cell else ["# (no header content)"])
    content.append("# %%")
    content.extend(rest if rest else ["# (no implementation content)"])
    out_path.write_text("\n".join(content).rstrip() + "\n", encoding="utf-8")


def _build_feedback(state: RunState) -> str:
    if state.last_parse_ok and state.pytest_returncode is not None:
        return textwrap.dedent(
            f"""
            Pytest return code: {state.pytest_returncode}
            Pytest stdout:
            {state.pytest_stdout.strip()}

            Pytest stderr:
            {state.pytest_stderr.strip()}
            """
        ).strip()
    if not state.last_parse_ok:
        return "Cursor output failed to parse. Output must include both solution.py and test_solution.py."
    return ""


def _prompt_path(config: AgentConfig, key: str, base_dir: Path) -> str:
    if key not in config.prompts:
        raise ValueError(f"Missing prompt key in config: {key}")
    rel = config.prompts[key]
    return str((base_dir / rel).resolve())


def run_poc_with_config(
    config_path: str,
    user_prompt: str,
    kpis: Optional[str],
    output_ipy_path: Optional[str] = None,
    max_iters: Optional[int] = None,
    workspace_dir: Optional[str] = None,
) -> str:
    """
    Runs a config-driven Cursor loop and exports a notebook-style .py file.
    Returns the path to the exported .py.
    """
    config = _load_config(config_path)
    base_dir = Path(config_path).resolve().parent
    if output_ipy_path:
        config.output_ipy_path = output_ipy_path
    if max_iters is not None:
        config.max_iters = int(max_iters)
    if workspace_dir:
        config.workspace_dir = workspace_dir

    ws = Path(config.workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)

    state = RunState(iteration=0)

    if kpis is None:
        kpi_template = _read_prompt_template(_prompt_path(config, "kpis", base_dir))
        kpi_prompt = _format_prompt(
            kpi_template,
            {
                "user_prompt": user_prompt.strip(),
                "pip_list": get_pip_list(),
                "iteration": "0",
                "feedback": "",
            },
        )
        kpis = call_cursor_cli(kpi_prompt, config).strip()

    for i in range(config.max_iters):
        state.iteration = i
        _archive_previous_versions(ws, i, config.output_files)
        feedback = _build_feedback(state) if i > 0 else ""
        prompt_key = "initial" if i == 0 else "repair"
        template = _read_prompt_template(_prompt_path(config, prompt_key, base_dir))
        prompt = _format_prompt(
            template,
            {
                "user_prompt": user_prompt.strip(),
                "kpis": kpis.strip(),
                "feedback": feedback,
                "iteration": str(i),
                "pip_list": get_pip_list(),
                "workspace_dir": str(ws.resolve()),
            },
        )

        raw = call_cursor_cli(prompt, config)
        state.last_cursor_raw = raw

        try:
            files = parse_cursor_output(raw)
            solution, tests = write_files(ws, files, config.output_files)
            state.solution_path = str(solution)
            state.tests_path = str(tests)
            state.last_parse_ok = True
        except Exception:
            state.last_parse_ok = False
            state.history.append(
                {
                    "iteration": i,
                    "parse_ok": state.last_parse_ok,
                    "pytest_returncode": None,
                    "failing_tests_estimate": None,
                    "ts": time.time(),
                }
            )
            if i >= config.max_iters - 1:
                state.converged = False
                state.stop_reason = "Max iterations reached (parse failures)."
                break
            continue

        rc, out, err, failing = run_pytest(ws, config.pytest_timeout_s)
        state.pytest_returncode = rc
        state.pytest_stdout = out
        state.pytest_stderr = err
        state.failing_tests_estimate = failing
        state.history.append(
            {
                "iteration": i,
                "parse_ok": state.last_parse_ok,
                "pytest_returncode": rc,
                "failing_tests_estimate": failing,
                "ts": time.time(),
            }
        )

        if rc == 0:
            state.converged = True
            state.stop_reason = "All tests passed (KPIs met)."
            break
        if i >= config.max_iters - 1:
            state.converged = False
            state.stop_reason = "Max iterations reached."

    # Diagnostics
    (ws / "history.json").write_text(json.dumps(state.history, indent=2), encoding="utf-8")
    (ws / "last_cursor_raw.txt").write_text(state.last_cursor_raw or "", encoding="utf-8")
    (ws / "last_pytest_stdout.txt").write_text(state.pytest_stdout or "", encoding="utf-8")
    (ws / "last_pytest_stderr.txt").write_text(state.pytest_stderr or "", encoding="utf-8")

    # Export
    sol_path = Path(state.solution_path) if state.solution_path else (ws / config.output_files[0])
    out_path = Path(config.output_ipy_path)
    export_to_ipy_py(sol_path, out_path)

    summary = {
        "converged": state.converged,
        "stop_reason": state.stop_reason,
        "iterations": state.iteration + 1,
        "workspace": str(ws.resolve()),
        "exported_ipy_py": str(out_path.resolve()),
    }
    (ws / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return str(out_path)
