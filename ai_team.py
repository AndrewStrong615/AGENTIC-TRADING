"""
ai_team.py
============================================================

AUTONOMOUS TRADING DEVELOPMENT TEAM
GROQ EDITION

BUG-FIRST + SAFE MODE + AUTOMATIC MERGE + AUTOMATIC PUSH

Workflow:

    1. Check that the live trader is stopped.
    2. Create an isolated Git worktree.
    3. Run the real FAST_BACKTEST.
    4. If it fails, Groq agents inspect/fix the bugs.
    5. Repeat bug fixing up to MAX_BUG_FIX_ROUNDS.
    6. Only after the bug gate passes, allow architecture work.
    7. Run the real backtest again.
    8. Check protected risk constants.
    9. Run final AI code review.
   10. Commit candidate.
   11. Verify candidate is still valid.
   12. Merge candidate into LIVE_DIR.
   13. Push the updated branch to origin.
   14. Failed candidates are NEVER merged.

IMPORTANT:

    GROQ_API_KEY must be set in the environment.

    This program never lets an LLM determine PASS/FAIL.
    PASS/FAIL comes from actual Python execution and the
    independent protected-risk guardrail.
"""

import os
import re
import subprocess
import sys
import time
from typing import Optional


from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool


# =====================================================================
# OPTIONAL PSUTIL
# =====================================================================

try:
    import psutil
except Exception:
    psutil = None


# =====================================================================
# 0. CONFIGURATION
# =====================================================================

LIVE_DIR = os.path.abspath(
    os.environ.get(
        "TRADING_LIVE_DIR",
        ".",
    )
)

WORK_ROOT = os.path.abspath(
    os.environ.get(
        "TRADING_WORK_ROOT",
        os.path.join("..", "trading-agent-work"),
    )
)

REMOTE_NAME = os.environ.get(
    "TRADING_GIT_REMOTE",
    "origin",
)

TARGET_BRANCH = os.environ.get(
    "TRADING_TARGET_BRANCH",
    "",
)

REVIEW_BRANCH_PREFIX = "agent-cycle"

PROTECTED_FILE = "Tradingcore.py"

LIVE_PROCESS_MARKERS = [
    "trader_v3.py",
]

CYCLE_SLEEP_SECONDS = int(
    os.environ.get(
        "CYCLE_SLEEP_SECONDS",
        "0",
    )
)

FAST_TEST_TIMEOUT_SECONDS = int(
    os.environ.get(
        "FAST_TEST_TIMEOUT_SECONDS",
        "900",
    )
)

MAX_BUG_FIX_ROUNDS = max(
    1,
    int(
        os.environ.get(
            "MAX_BUG_FIX_ROUNDS",
            "3",
        )
    ),
)

MAX_ARCHITECTURE_ROUNDS = max(
    0,
    int(
        os.environ.get(
            "MAX_ARCHITECTURE_ROUNDS",
            "1",
        )
    ),
)


# =====================================================================
# 1. PROTECTED RISK CONSTANTS
# =====================================================================

PROTECTED_CONSTANTS = [
    "CONFIDENCE_THRESHOLD",
    "MAX_POSITION_PER_SYMBOL",
    "MIN_SECONDS_BETWEEN_ORDERS",
    "ATR_STOP_MULTIPLIER",
    "ATR_TARGET_MULTIPLIER",
    "MIN_STOP_PCT",
    "MAX_STOP_PCT",
    "DAILY_LOSS_LIMIT_PCT",
    "MAX_QTY_PER_ORDER",
    "KELLY_FRACTION_MULTIPLIER",
    "MAX_KELLY_FRACTION",
    "MAX_POSITION_PCT",
    "RISK_PER_TRADE_PCT",
    "FIXED_STOP_PCT",
    "FIXED_TARGET_PCT",
]


# =====================================================================
# 2. GROQ CONFIGURATION
# =====================================================================

GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:
    print()
    print("=" * 70)
    print("ERROR: GROQ_API_KEY IS NOT SET")
    print("=" * 70)
    print()
    print("You are using Command Prompt.")
    print()
    print("Set it with:")
    print()
    print("    set GROQ_API_KEY=YOUR_NEW_KEY")
    print()
    print("Or permanently:")
    print()
    print('    setx GROQ_API_KEY "YOUR_NEW_KEY"')
    print()
    print("Then open a NEW Command Prompt.")
    print()
    sys.exit(1)


# ---------------------------------------------------------------------
# Groq model
# ---------------------------------------------------------------------
#
# CrewAI uses LiteLLM underneath for provider/model routing.
#
# You can change this environment variable without editing the file:
#
#     set GROQ_MODEL=groq/llama-3.3-70b-versatile
#
# ---------------------------------------------------------------------

GROQ_MODEL = os.environ.get(
    "GROQ_MODEL",
    "groq/llama-3.3-70b-versatile",
)

groq_llm = LLM(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
)


# =====================================================================
# 3. SUBPROCESS HELPER
# =====================================================================

def _run(
    cmd,
    cwd=None,
    env=None,
    timeout=None,
):
    """
    Run a subprocess safely.

    Returns:
        (returncode, stdout, stderr)
    """

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        return (
            proc.returncode,
            proc.stdout or "",
            proc.stderr or "",
        )

    except subprocess.TimeoutExpired as exc:

        stdout = exc.stdout or ""

        stderr = (
            f"TIMEOUT after {timeout}s\n"
            f"{exc}"
        )

        return (
            -1,
            stdout,
            stderr,
        )

    except KeyboardInterrupt:
        raise

    except Exception as exc:

        return (
            -1,
            "",
            str(exc),
        )


# =====================================================================
# 4. LIVE TRADER CHECK
# =====================================================================

def live_trader_is_running():
    """
    Safely determine whether the live trader appears to be running.

    If process inspection cannot be trusted, return True.
    """

    if psutil is None:

        print(
            "WARNING: psutil is not installed."
        )

        print(
            "Cannot safely verify trader state."
        )

        print(
            "Install with:"
        )

        print(
            "    pip install psutil"
        )

        return True

    try:

        current_pid = os.getpid()

        for process in psutil.process_iter(
            ["pid", "cmdline"]
        ):

            try:

                pid = process.info.get(
                    "pid"
                )

                if pid == current_pid:
                    continue

                cmdline_parts = (
                    process.info.get(
                        "cmdline"
                    )
                    or []
                )

                cmdline = " ".join(
                    str(part)
                    for part in cmdline_parts
                )

                for marker in LIVE_PROCESS_MARKERS:

                    if marker.lower() in cmdline.lower():

                        return True

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess,
            ):

                continue

    except Exception:

        return True

    return False


# =====================================================================
# 5. GIT HELPERS
# =====================================================================

def git_current_branch(
    directory,
):
    """
    Return current branch.
    """

    rc, stdout, stderr = _run(
        [
            "git",
            "-C",
            directory,
            "branch",
            "--show-current",
        ]
    )

    if rc != 0:

        raise RuntimeError(
            f"Unable to determine Git branch:\n"
            f"{stderr}"
        )

    return stdout.strip()


def git_has_uncommitted_changes(
    directory,
):
    """
    Determine whether the working tree has uncommitted changes.
    """

    rc, stdout, stderr = _run(
        [
            "git",
            "-C",
            directory,
            "status",
            "--porcelain",
        ]
    )

    if rc != 0:

        raise RuntimeError(
            f"Unable to check Git status:\n"
            f"{stderr}"
        )

    return bool(
        stdout.strip()
    )


def determine_target_branch():
    """
    Determine the branch that automatic merging should target.

    Priority:

        TRADING_TARGET_BRANCH
        current branch of LIVE_DIR
    """

    if TARGET_BRANCH.strip():

        return TARGET_BRANCH.strip()

    branch = git_current_branch(
        LIVE_DIR
    )

    if not branch:

        raise RuntimeError(
            "LIVE_DIR is currently in detached HEAD state.\n"
            "Set TRADING_TARGET_BRANCH explicitly."
        )

    return branch


def prepare_work_branch(
    cycle_count,
):
    """
    Create a fresh isolated Git worktree.
    """

    rc, _, err = _run(
        [
            "git",
            "-C",
            LIVE_DIR,
            "rev-parse",
            "--is-inside-work-tree",
        ]
    )

    if rc != 0:

        raise RuntimeError(
            f"{LIVE_DIR} is not a Git repository.\n"
            f"{err.strip()}"
        )

    rc, head, err = _run(
        [
            "git",
            "-C",
            LIVE_DIR,
            "rev-parse",
            "HEAD",
        ]
    )

    if rc != 0:

        raise RuntimeError(
            "Repository has no valid HEAD.\n"
            "Create a baseline commit first."
        )

    branch_name = (
        f"{REVIEW_BRANCH_PREFIX}-"
        f"{cycle_count}-"
        f"{int(time.time())}"
    )

    work_dir = os.path.join(
        WORK_ROOT,
        branch_name,
    )

    os.makedirs(
        WORK_ROOT,
        exist_ok=True,
    )

    if os.path.exists(work_dir):

        raise RuntimeError(
            f"Worktree already exists:\n"
            f"{work_dir}"
        )

    rc, _, err = _run(
        [
            "git",
            "-C",
            LIVE_DIR,
            "worktree",
            "add",
            "-b",
            branch_name,
            work_dir,
            "HEAD",
        ]
    )

    if rc != 0:

        raise RuntimeError(
            "Failed to create worktree:\n"
            f"{err.strip()}"
        )

    return (
        branch_name,
        work_dir,
        head.strip(),
    )


# =====================================================================
# 6. PROTECTED CONSTANT GUARDRAIL
# =====================================================================

def read_protected_constants(
    base_dir,
):
    """
    Read protected constants without importing Tradingcore.py.
    """

    path = os.path.join(
        base_dir,
        PROTECTED_FILE,
    )

    if not os.path.exists(path):

        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:

            text = file.read()

    except Exception:

        return {}

    values = {}

    for name in PROTECTED_CONSTANTS:

        pattern = (
            rf"^\s*"
            rf"{re.escape(name)}"
            rf"\s*=\s*(.+?)"
            rf"\s*$"
        )

        match = re.search(
            pattern,
            text,
            re.MULTILINE,
        )

        values[name] = (
            match.group(1).strip()
            if match
            else None
        )

    return values


def guardrail_check(
    live_dir,
    work_dir,
):
    """
    Compare protected constants.
    """

    live_values = read_protected_constants(
        live_dir
    )

    work_values = read_protected_constants(
        work_dir
    )

    diffs = {}

    for name in PROTECTED_CONSTANTS:

        live_value = live_values.get(
            name
        )

        work_value = work_values.get(
            name
        )

        if live_value != work_value:

            diffs[name] = (
                live_value,
                work_value,
            )

    return (
        len(diffs) == 0,
        diffs,
    )


# =====================================================================
# 7. PYTHON COMPILE GATE
# =====================================================================

def run_python_compile_gate(
    work_dir,
):
    """
    Compile every Python file except Tradingcore.py.
    """

    python_files = []

    for root, dirs, files in os.walk(
        work_dir
    ):

        dirs[:] = [
            directory
            for directory in dirs
            if directory != ".git"
            and directory != "__pycache__"
        ]

        for filename in files:

            if (
                filename.endswith(".py")
                and filename != PROTECTED_FILE
            ):

                python_files.append(
                    os.path.join(
                        root,
                        filename,
                    )
                )

    if not python_files:

        return (
            False,
            -1,
            "",
            "No Python files were found.",
        )

    errors = []
    compiled_count = 0

    for path in sorted(
        python_files
    ):

        relative = os.path.relpath(
            path,
            work_dir,
        )

        rc, stdout, stderr = _run(
            [
                sys.executable,
                "-m",
                "py_compile",
                relative,
            ],
            cwd=work_dir,
            timeout=120,
        )

        if rc != 0:

            errors.append(
                f"FILE: {relative}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}"
            )

        else:

            compiled_count += 1

    if errors:

        return (
            False,
            1,
            f"Successfully compiled: "
            f"{compiled_count}",
            "\n\n".join(errors),
        )

    return (
        True,
        0,
        f"Successfully compiled "
        f"{compiled_count} Python files.",
        "",
    )


# =====================================================================
# 8. REAL BACKTEST
# =====================================================================

def run_real_backtest_gate(
    work_dir,
):
    """
    Run actual backtest.py.

    FAST_BACKTEST=1 is enabled.

    Python exit code determines PASS/FAIL.
    """

    (
        compile_passed,
        compile_rc,
        compile_stdout,
        compile_stderr,
    ) = run_python_compile_gate(
        work_dir
    )

    if not compile_passed:

        return (
            False,
            compile_rc,
            compile_stdout,
            (
                "PYTHON COMPILE GATE FAILED\n\n"
                f"{compile_stderr}"
            ),
        )

    env = {
        **os.environ,
        "FAST_BACKTEST": "1",
    }

    returncode, stdout, stderr = _run(
        [
            sys.executable,
            "backtest.py",
        ],
        cwd=work_dir,
        env=env,
        timeout=FAST_TEST_TIMEOUT_SECONDS,
    )

    passed = (
        returncode == 0
    )

    return (
        passed,
        returncode,
        stdout[-12000:],
        stderr[-12000:],
    )


# =====================================================================
# 9. SAVE BACKTEST RESULT
# =====================================================================

def save_backtest_result(
    work_dir,
    round_number,
    passed,
    returncode,
    stdout,
    stderr,
):
    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    path = os.path.join(
        work_dir,
        f"BUG_TEST_RESULT_{round_number}.md",
    )

    content = (
        "# Bug Test Result\n\n"
        "## VERIFIED STATUS\n\n"
        f"{status}\n\n"
        "## Exit Code\n\n"
        f"{returncode}\n\n"
        "## STDOUT\n\n"
        "```text\n"
        f"{stdout or '(none)'}\n"
        "```\n\n"
        "## STDERR\n\n"
        "```text\n"
        f"{stderr or '(none)'}\n"
        "```\n"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(content)

    return path


# =====================================================================
# 10. SAFE PATH
# =====================================================================

def _safe_path(
    work_dir,
    relative_path,
):
    """
    Prevent path traversal.
    """

    if not relative_path:

        raise ValueError(
            "A file path is required."
        )

    full = os.path.abspath(
        os.path.join(
            work_dir,
            relative_path,
        )
    )

    root = os.path.abspath(
        work_dir
    )

    try:

        common = os.path.commonpath(
            [
                root,
                full,
            ]
        )

    except ValueError:

        raise ValueError(
            f"Invalid path: {relative_path}"
        )

    if common != root:

        raise ValueError(
            "Refusing to access path outside "
            f"project: {relative_path}"
        )

    return full


# =====================================================================
# 11. TEST TOOL
# =====================================================================

def make_test_tool(
    work_dir,
):

    @tool("run_fast_backtest")
    def run_fast_backtest() -> str:
        """
        Run the actual FAST_BACKTEST.
        """

        (
            passed,
            returncode,
            stdout,
            stderr,
        ) = run_real_backtest_gate(
            work_dir
        )

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        return (
            f"STATUS: {status}\n"
            f"EXIT CODE: {returncode}\n\n"
            f"STDOUT:\n{stdout}\n\n"
            f"STDERR:\n{stderr}"
        )

    return run_fast_backtest


# =====================================================================
# 12. PROJECT TOOLS
# =====================================================================

def make_tools(
    work_dir,
):

    @tool("list_project_files")
    def list_project_files() -> str:
        """
        List project files.
        """

        names = []

        for root, dirs, files in os.walk(
            work_dir
        ):

            dirs[:] = [
                directory
                for directory in dirs
                if directory != ".git"
                and directory != "__pycache__"
            ]

            for filename in files:

                names.append(
                    os.path.relpath(
                        os.path.join(
                            root,
                            filename,
                        ),
                        work_dir,
                    )
                )

        return "\n".join(
            sorted(names)
        )


    @tool("read_project_file")
    def read_project_file(
        relative_path: str,
    ) -> str:
        """
        Read a project file.
        """

        try:

            path = _safe_path(
                work_dir,
                relative_path,
            )

        except ValueError as exc:

            return f"ERROR: {exc}"

        if not os.path.exists(path):

            return (
                f"ERROR: {relative_path} "
                "does not exist."
            )

        if not os.path.isfile(path):

            return (
                f"ERROR: {relative_path} "
                "is not a file."
            )

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
                errors="replace",
            ) as file:

                return file.read()

        except Exception as exc:

            return (
                f"ERROR reading "
                f"{relative_path}: {exc}"
            )


    @tool("write_project_file")
    def write_project_file(
        relative_path: str,
        content: str,
    ) -> str:
        """
        Write a project file.

        Tradingcore.py is permanently protected.
        """

        normalized = os.path.normpath(
            relative_path
        )

        if (
            os.path.basename(
                normalized
            ).lower()
            == PROTECTED_FILE.lower()
        ):

            return (
                f"REFUSED: {PROTECTED_FILE} "
                "is protected and cannot be "
                "modified by an agent."
            )

        try:

            path = _safe_path(
                work_dir,
                normalized,
            )

        except ValueError as exc:

            return f"REFUSED: {exc}"

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True,
        )

        try:

            with open(
                path,
                "w",
                encoding="utf-8",
            ) as file:

                file.write(content)

        except Exception as exc:

            return (
                f"ERROR writing "
                f"{relative_path}: {exc}"
            )

        return (
            f"Wrote {len(content)} "
            f"characters to "
            f"{relative_path}."
        )

    return [
        list_project_files,
        read_project_file,
        write_project_file,
    ]


# =====================================================================
# 13. BUG CREW
# =====================================================================

def build_bug_crew(
    work_dir,
    test_result_text="",
):

    tools = make_tools(
        work_dir
    )

    test_tool = make_test_tool(
        work_dir
    )

    bug_tools = tools + [
        test_tool
    ]


    bug_detector = Agent(

        role="Senior Trading Software Bug Detective",

        goal=(
            "Find actual confirmed bugs in the trading bot. "
            "Prioritize runtime failures, KeyErrors, missing "
            "symbols, malformed market data, incorrect data flow, "
            "state errors, and reproducible backtest failures."
        ),

        backstory=(
            "You are a senior Python debugging engineer working "
            "on a quantitative trading software project. "
            "You are evidence-driven. You never invent bugs. "
            "You inspect the actual source and execution output. "
            "Correctness comes before optimization."
        ),

        tools=bug_tools,

        verbose=True,

        allow_delegation=False,

        llm=groq_llm,
    )


    bug_fixer = Agent(

        role="Senior Trading Bot Bug Fix Engineer",

        goal=(
            "Fix confirmed bugs with the smallest safe change "
            "possible and verify the fix with the real backtest."
        ),

        backstory=(
            "You are an expert Python engineer specializing in "
            "pandas, NumPy, market data, Alpaca integrations, "
            "backtesting systems, and stateful applications. "
            "You never modify Tradingcore.py."
        ),

        tools=bug_tools,

        verbose=True,

        allow_delegation=False,

        llm=groq_llm,
    )


    bug_validator = Agent(

        role="Automated Bug Validation Engineer",

        goal=(
            "Verify that bug fixes actually solve the reported "
            "failure using real execution evidence."
        ),

        backstory=(
            "You are skeptical and evidence-driven. A fix is not "
            "considered successful because the code looks correct. "
            "You run the real FAST_BACKTEST and inspect its output."
        ),

        tools=bug_tools,

        verbose=True,

        allow_delegation=False,

        llm=groq_llm,
    )


    task_detect = Task(

        description=(
            "BUG-FIRST TASK.\n\n"

            "Inspect the isolated project thoroughly.\n\n"

            "Use list_project_files and read_project_file.\n\n"

            "The previous real execution result is:\n\n"
            f"{test_result_text}\n\n"

            "Find confirmed bugs.\n\n"

            "Prioritize:\n"
            "1. crashes\n"
            "2. syntax/import errors\n"
            "3. KeyError/IndexError/TypeError\n"
            "4. malformed data\n"
            "5. missing-symbol handling\n"
            "6. incorrect runtime behavior\n"
            "7. backtest correctness\n\n"

            "Do NOT perform architecture work.\n"
            "Do NOT optimize.\n"
            "Do NOT change risk controls.\n\n"

            "Write BUG_REPORT.md."
        ),

        expected_output=(
            "BUG_REPORT.md containing prioritized bugs "
            "and evidence."
        ),

        agent=bug_detector,
    )


    task_fix = Task(

        description=(
            "Read BUG_REPORT.md.\n\n"

            "Inspect the actual execution failure.\n\n"

            "Fix the highest-priority confirmed bugs.\n\n"

            "Rules:\n"
            "- Never modify Tradingcore.py.\n"
            "- Never change protected risk constants.\n"
            "- Do not add unrelated features.\n"
            "- Do not perform speculative rewrites.\n"
            "- Preserve existing behavior.\n"
            "- Use write_project_file for modifications.\n"
            "- Run run_fast_backtest after fixing."
        ),

        expected_output=(
            "A list of fixes and the actual "
            "FAST_BACKTEST result."
        ),

        agent=bug_fixer,
    )


    task_validate = Task(

        description=(
            "Validate the changes.\n\n"

            "Inspect changed files.\n\n"

            "Run run_fast_backtest.\n\n"

            "If it fails, write the exact remaining failure "
            "to BUG_VALIDATION.md.\n\n"

            "Do not declare success without execution evidence."
        ),

        expected_output=(
            "BUG_VALIDATION.md containing the actual "
            "validation result."
        ),

        agent=bug_validator,
    )


    return Crew(

        agents=[
            bug_detector,
            bug_fixer,
            bug_validator,
        ],

        tasks=[
            task_detect,
            task_fix,
            task_validate,
        ],

        process=Process.sequential,

        verbose=True,
    )


# =====================================================================
# 14. ARCHITECTURE CREW
# =====================================================================

def build_architecture_crew(
    work_dir,
):

    tools = make_tools(
        work_dir
    )


    architect = Agent(

        role="Lead Quantitative Software Architect",

        goal=(
            "Improve architecture, modularity, maintainability, "
            "data-pipeline quality, and performance only after "
            "the bug gate has passed."
        ),

        backstory=(
            "You are a senior quantitative software architect. "
            "You understand Python trading systems, market-data "
            "pipelines, indicators, state machines, and backtests. "
            "Correctness always comes before optimization."
        ),

        tools=tools,

        verbose=True,

        allow_delegation=False,

        llm=groq_llm,
    )


    developer = Agent(

        role="Senior Algorithmic Trading Developer",

        goal=(
            "Implement only safe, targeted architectural "
            "improvements."
        ),

        backstory=(
            "You are a senior Python developer. "
            "You make minimal, testable changes. "
            "You never modify Tradingcore.py."
        ),

        tools=tools,

        verbose=True,

        allow_delegation=False,

        llm=groq_llm,
    )


    task_plan = Task(

        description=(
            "The bug gate has passed.\n\n"

            "Inspect the entire project.\n\n"

            "Identify only worthwhile improvements to:\n"
            "- architecture\n"
            "- maintainability\n"
            "- modularity\n"
            "- data handling\n"
            "- performance\n\n"

            "Do not modify risk controls.\n"
            "Do not modify Tradingcore.py.\n"
            "Do not invent bugs.\n\n"

            "Write TRADING_ARCHITECTURE.md."
        ),

        expected_output=(
            "TRADING_ARCHITECTURE.md containing a focused plan."
        ),

        agent=architect,
    )


    task_implement = Task(

        description=(
            "Read TRADING_ARCHITECTURE.md.\n\n"

            "Implement only the highest-value safe improvement.\n\n"

            "Never modify Tradingcore.py.\n"
            "Never change protected risk constants.\n"
            "Avoid unnecessary rewrites.\n"
            "Use write_project_file."
        ),

        expected_output=(
            "List of changed files and reasons."
        ),

        agent=developer,
    )


    return Crew(

        agents=[
            architect,
            developer,
        ],

        tasks=[
            task_plan,
            task_implement,
        ],

        process=Process.sequential,

        verbose=True,
    )


# =====================================================================
# 15. REVIEW CREW
# =====================================================================

def build_review_crew(
    work_dir,
):

    tools = make_tools(
        work_dir
    )


    reviewer = Agent(

        role="Senior Code Review and QA Analyst",

        goal=(
            "Review the candidate honestly and identify anything "
            "that should be investigated before automatic merge."
        ),

        backstory=(
            "You are an experienced code reviewer. "
            "You do not determine PASS/FAIL. "
            "Python execution and independent guardrails do that."
        ),

        tools=tools,

        verbose=True,

        allow_delegation=False,

        llm=groq_llm,
    )


    task = Task(

        description=(
            "Review the candidate worktree.\n\n"

            "Inspect:\n"
            "- BUG_REPORT.md\n"
            "- BUG_VALIDATION.md\n"
            "- TRADING_ARCHITECTURE.md\n"
            "- changed source files\n"
            "- actual backtest results\n\n"

            "Write REVIEW_COMMENTARY.md.\n\n"

            "Do not claim PASS or FAIL.\n"
            "Identify concerns a human should know about."
        ),

        expected_output=(
            "REVIEW_COMMENTARY.md containing review notes."
        ),

        agent=reviewer,
    )


    return Crew(

        agents=[
            reviewer,
        ],

        tasks=[
            task,
        ],

        process=Process.sequential,

        verbose=True,
    )


# =====================================================================
# 16. AUDIT REPORT
# =====================================================================

def write_audit_report(
    work_dir,
    guard_passed,
    guard_diffs,
    test_passed,
    returncode,
    stdout,
    stderr,
    commentary="",
    bug_rounds=0,
    architecture_ran=False,
):

    verified_status = (
        "PASS"
        if (
            guard_passed
            and test_passed
        )
        else "FAIL"
    )

    lines = [

        "# Risk Audit Report",

        "",

        f"**VERIFIED_STATUS: {verified_status}**",

        "",

        (
            "PASS/FAIL was determined by actual Python execution "
            "and protected-risk comparison."
        ),

        "",

        "## Configuration",

        "",

        f"- Groq model: `{GROQ_MODEL}`",

        f"- Target branch: `{determine_target_branch()}`",

        "",

        "## Development",

        "",

        f"- Bug-fix rounds: {bug_rounds}",

        f"- Architecture phase: {architecture_ran}",

        "",

        "## Protected Risk Guardrail",

        "",
    ]

    if guard_passed:

        lines.append(
            "PASS — protected risk constants unchanged."
        )

    else:

        lines.append(
            "FAIL — protected risk constants changed."
        )

        for name, (
            live_value,
            work_value,
        ) in guard_diffs.items():

            lines.append(
                f"- `{name}`: "
                f"`{live_value}` -> "
                f"`{work_value}`"
            )

    lines += [

        "",

        "## Final Real Backtest",

        "",

        f"Exit code: {returncode}",

        "",

        "### STDOUT",

        "",

        "```text",

        stdout or "(none)",

        "```",

        "",

        "### STDERR",

        "",

        "```text",

        stderr or "(none)",

        "```",
    ]

    if commentary:

        lines += [

            "",

            "## Agent Commentary",

            "",

            commentary,
        ]

    path = os.path.join(
        work_dir,
        "RISK_AUDIT_REPORT.md",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(lines)
        )

    return verified_status


# =====================================================================
# 17. COMMIT
# =====================================================================

def commit_worktree(
    work_dir,
    message,
):

    rc, _, err = _run(
        [
            "git",
            "-C",
            work_dir,
            "add",
            "-A",
        ]
    )

    if rc != 0:

        return (
            rc,
            "",
            f"git add failed:\n{err}",
        )

    rc, stdout, stderr = _run(
        [
            "git",
            "-C",
            work_dir,
            "commit",
            "-m",
            message,
        ]
    )

    if rc != 0:

        combined = (
            f"{stdout}\n{stderr}"
        ).lower()

        if "nothing to commit" in combined:

            return (
                0,
                stdout,
                stderr,
            )

    return (
        rc,
        stdout,
        stderr,
    )


# =====================================================================
# 18. AUTOMATIC MERGE + PUSH
# =====================================================================

def automatic_merge_and_push(
    branch_name,
    candidate_commit,
):
    """
    Merge candidate branch into LIVE_DIR and push it.

    IMPORTANT:

    This function is called ONLY after:
        - compile gate passed
        - real backtest passed
        - risk guardrail passed
        - candidate was committed
    """

    target_branch = determine_target_branch()

    print()
    print("=" * 70)
    print("AUTOMATIC MERGE")
    print("=" * 70)

    print(
        f"Target branch: {target_branch}"
    )

    print(
        f"Candidate branch: {branch_name}"
    )

    # ---------------------------------------------------------------
    # Verify live repository is clean.
    # ---------------------------------------------------------------

    if git_has_uncommitted_changes(
        LIVE_DIR
    ):

        print()
        print(
            "❌ LIVE REPOSITORY HAS UNCOMMITTED CHANGES."
        )

        print(
            "Automatic merge cancelled."
        )

        return False

    # ---------------------------------------------------------------
    # Verify live branch.
    # ---------------------------------------------------------------

    current_branch = git_current_branch(
        LIVE_DIR
    )

    if current_branch != target_branch:

        print(
            f"Switching LIVE_DIR from "
            f"{current_branch} to "
            f"{target_branch}"
        )

        rc, _, stderr = _run(
            [
                "git",
                "-C",
                LIVE_DIR,
                "checkout",
                target_branch,
            ]
        )

        if rc != 0:

            print(
                "❌ Could not checkout target branch."
            )

            print(stderr)

            return False

    # ---------------------------------------------------------------
    # Make sure candidate commit exists.
    # ---------------------------------------------------------------

    rc, _, stderr = _run(
        [
            "git",
            "-C",
            LIVE_DIR,
            "cat-file",
            "-e",
            f"{candidate_commit}^{{commit}}",
        ]
    )

    if rc != 0:

        print(
            "❌ Candidate commit could not be verified."
        )

        print(stderr)

        return False

    # ---------------------------------------------------------------
    # Fetch latest remote state.
    # ---------------------------------------------------------------

    print(
        "Fetching latest remote state..."
    )

    rc, stdout, stderr = _run(
        [
            "git",
            "-C",
            LIVE_DIR,
            "fetch",
            REMOTE_NAME,
        ]
    )

    if rc != 0:

        print(
            "❌ Git fetch failed."
        )

        print(stderr)

        return False

    # ---------------------------------------------------------------
    # Make sure local target hasn't diverged unexpectedly.
    # ---------------------------------------------------------------

    rc, _, stderr = _run(
        [
            "git",
            "-C",
            LIVE_DIR,
            "merge-base",
            "--is-ancestor",
            "HEAD",
            f"{REMOTE_NAME}/{target_branch}",
        ]
    )

    if rc != 0:

        print()
        print(
            "⚠️ Remote branch changed or local branch is not "
            "behind the remote."
        )

        print(
            "Automatic merge cancelled to prevent overwriting "
            "new remote work."
        )

        return False

    # ---------------------------------------------------------------
    # Merge candidate.
    # ---------------------------------------------------------------

    print(
        "Merging candidate..."
    )

    rc, stdout, stderr = _run(
        [
            "git",
            "-C",
            LIVE_DIR,
            "merge",
            "--no-ff",
            branch_name,
            "-m",
            (
                f"Automatic merge: "
                f"{branch_name}"
            ),
        ]
    )

    if rc != 0:

        print()
        print(
            "❌ AUTOMATIC MERGE FAILED."
        )

        print(stderr)

        print(
            "Aborting merge..."
        )

        _run(
            [
                "git",
                "-C",
                LIVE_DIR,
                "merge",
                "--abort",
            ]
        )

        return False

    print(
        "✅ Candidate merged successfully."
    )

    # ---------------------------------------------------------------
    # Push.
    # ---------------------------------------------------------------

    print()
    print(
        "Pushing merged branch..."
    )

    rc, stdout, stderr = _run(
        [
            "git",
            "-C",
            LIVE_DIR,
            "push",
            REMOTE_NAME,
            target_branch,
        ]
    )

    if rc != 0:

        print()
        print(
            "❌ PUSH FAILED."
        )

        print(stderr)

        print()
        print(
            "The merge exists locally."
        )

        print(
            "DO NOT run another automatic cycle until "
            "the Git state is checked."
        )

        return False

    print()
    print(
        "✅ PUSH SUCCESSFUL."
    )

    print(
        f"Remote: {REMOTE_NAME}"
    )

    print(
        f"Branch: {target_branch}"
    )

    return True


# =====================================================================
# 19. CANDIDATE DIFF
# =====================================================================

def get_candidate_diff(
    live_dir,
    branch_name,
):

    rc, stdout, stderr = _run(
        [
            "git",
            "-C",
            live_dir,
            "diff",
            "HEAD",
            branch_name,
            "--",
            ".",
        ]
    )

    if rc != 0:

        return (
            "Unable to obtain diff:\n"
            f"{stderr}"
        )

    return stdout


# =====================================================================
# 20. FILE READER
# =====================================================================

def read_text_file(
    path,
):

    if not os.path.exists(path):

        return ""

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:

            return file.read()

    except Exception:

        return ""


# =====================================================================
# 21. COMPLETE CYCLE
# =====================================================================

def run_cycle(
    cycle_count,
):

    branch_name = None
    work_dir = None

    test_passed = False
    returncode = None
    stdout = ""
    stderr = ""

    try:

        # =============================================================
        # SAFETY CHECK
        # =============================================================

        if live_trader_is_running():

            print(
                "⏸ Live trader appears to be running."
            )

            print(
                "Skipping this cycle."
            )

            return

        # =============================================================
        # CREATE WORKTREE
        # =============================================================

        (
            branch_name,
            work_dir,
            baseline_commit,
        ) = prepare_work_branch(
            cycle_count
        )

        print(
            f"Isolated branch: {branch_name}"
        )

        print(
            f"Worktree: {work_dir}"
        )

        # =============================================================
        # INITIAL BACKTEST
        # =============================================================

        print()
        print("=" * 70)
        print("INITIAL REAL BACKTEST")
        print("=" * 70)

        (
            initial_passed,
            initial_returncode,
            initial_stdout,
            initial_stderr,
        ) = run_real_backtest_gate(
            work_dir
        )

        save_backtest_result(
            work_dir,
            "initial",
            initial_passed,
            initial_returncode,
            initial_stdout,
            initial_stderr,
        )

        if initial_passed:

            print(
                "✅ Initial backtest passed."
            )

        else:

            print(
                "❌ Initial backtest failed."
            )

        bug_test_output = (
            f"Initial backtest exit code: "
            f"{initial_returncode}\n\n"
            f"STDOUT:\n"
            f"{initial_stdout}\n\n"
            f"STDERR:\n"
            f"{initial_stderr}"
        )

        # =============================================================
        # BUG LOOP
        # =============================================================

        bug_gate_passed = initial_passed

        bug_rounds_completed = 0

        for bug_round in range(
            1,
            MAX_BUG_FIX_ROUNDS + 1,
        ):

            if bug_gate_passed:
                break

            bug_rounds_completed = bug_round

            print()
            print("=" * 70)

            print(
                f"🐛 BUG-FIX ROUND "
                f"{bug_round}/"
                f"{MAX_BUG_FIX_ROUNDS}"
            )

            print("=" * 70)

            bug_crew = build_bug_crew(
                work_dir,
                bug_test_output,
            )

            bug_crew.kickoff()

            print(
                "Bug crew completed."
            )

            print(
                "Running real FAST_BACKTEST..."
            )

            (
                test_passed,
                returncode,
                stdout,
                stderr,
            ) = run_real_backtest_gate(
                work_dir
            )

            save_backtest_result(
                work_dir,
                bug_round,
                test_passed,
                returncode,
                stdout,
                stderr,
            )

            bug_test_output = (
                f"Bug round {bug_round} result.\n\n"
                f"Exit code: {returncode}\n\n"
                f"STDOUT:\n{stdout}\n\n"
                f"STDERR:\n{stderr}"
            )

            if test_passed:

                print(
                    "✅ BUG GATE PASSED."
                )

                bug_gate_passed = True

                break

            print(
                "❌ Backtest still failing."
            )

        # =============================================================
        # BUG GATE FAILED
        # =============================================================

        if not bug_gate_passed:

            print()
            print("=" * 70)
            print("❌ BUG GATE FAILED")
            print("=" * 70)

            print(
                "Architecture work will NOT run."
            )

            guard_passed, guard_diffs = (
                guardrail_check(
                    LIVE_DIR,
                    work_dir,
                )
            )

            commentary = read_text_file(
                os.path.join(
                    work_dir,
                    "REVIEW_COMMENTARY.md",
                )
            )

            status = write_audit_report(
                work_dir=work_dir,
                guard_passed=guard_passed,
                guard_diffs=guard_diffs,
                test_passed=False,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                commentary=commentary,
                bug_rounds=bug_rounds_completed,
                architecture_ran=False,
            )

            commit_worktree(
                work_dir,
                f"Agent cycle #{cycle_count}: {status}",
            )

            print(
                f"VERIFIED_STATUS: {status}"
            )

            print(
                f"Candidate preserved: {work_dir}"
            )

            print(
                "❌ Nothing will be merged."
            )

            return

        # =============================================================
        # ARCHITECTURE
        # =============================================================

        architecture_ran = False

        if MAX_ARCHITECTURE_ROUNDS > 0:

            architecture_ran = True

            for architecture_round in range(
                1,
                MAX_ARCHITECTURE_ROUNDS + 1,
            ):

                print()
                print("=" * 70)

                print(
                    f"🏗 ARCHITECTURE ROUND "
                    f"{architecture_round}/"
                    f"{MAX_ARCHITECTURE_ROUNDS}"
                )

                print("=" * 70)

                architecture_crew = (
                    build_architecture_crew(
                        work_dir
                    )
                )

                architecture_crew.kickoff()

                print(
                    "Architecture work completed."
                )

                print(
                    "Running real backtest..."
                )

                (
                    test_passed,
                    returncode,
                    stdout,
                    stderr,
                ) = run_real_backtest_gate(
                    work_dir
                )

                save_backtest_result(
                    work_dir,
                    f"architecture-{architecture_round}",
                    test_passed,
                    returncode,
                    stdout,
                    stderr,
                )

                if not test_passed:

                    print(
                        "❌ Architecture broke the backtest."
                    )

                    print(
                        "Candidate will NOT be merged."
                    )

                    break

                print(
                    "✅ Architecture passed."
                )

        # =============================================================
        # FINAL BACKTEST
        # =============================================================

        if architecture_ran and not test_passed:

            final_status = "FAIL"

            guard_passed, guard_diffs = (
                guardrail_check(
                    LIVE_DIR,
                    work_dir,
                )
            )

            commentary = read_text_file(
                os.path.join(
                    work_dir,
                    "REVIEW_COMMENTARY.md",
                )
            )

            write_audit_report(
                work_dir=work_dir,
                guard_passed=guard_passed,
                guard_diffs=guard_diffs,
                test_passed=False,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                commentary=commentary,
                bug_rounds=bug_rounds_completed,
                architecture_ran=True,
            )

            commit_worktree(
                work_dir,
                f"Agent cycle #{cycle_count}: {final_status}",
            )

            print(
                "❌ Candidate preserved."
            )

            return

        # =============================================================
        # FINAL REVIEW
        # =============================================================

        print()
        print(
            "Running final Groq code review..."
        )

        review_crew = build_review_crew(
            work_dir
        )

        review_crew.kickoff()

        commentary = read_text_file(
            os.path.join(
                work_dir,
                "REVIEW_COMMENTARY.md",
            )
        )

        # =============================================================
        # FINAL BACKTEST AGAIN
        # =============================================================

        print()
        print(
            "Running FINAL real FAST_BACKTEST..."
        )

        (
            test_passed,
            returncode,
            stdout,
            stderr,
        ) = run_real_backtest_gate(
            work_dir
        )

        save_backtest_result(
            work_dir,
            "final",
            test_passed,
            returncode,
            stdout,
            stderr,
        )

        if not test_passed:

            print(
                "❌ FINAL BACKTEST FAILED."
            )

            print(
                "Candidate will NOT be merged."
            )

            guard_passed, guard_diffs = (
                guardrail_check(
                    LIVE_DIR,
                    work_dir,
                )
            )

            status = write_audit_report(
                work_dir=work_dir,
                guard_passed=guard_passed,
                guard_diffs=guard_diffs,
                test_passed=False,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                commentary=commentary,
                bug_rounds=bug_rounds_completed,
                architecture_ran=architecture_ran,
            )

            commit_worktree(
                work_dir,
                f"Agent cycle #{cycle_count}: {status}",
            )

            return

        # =============================================================
        # RISK GUARDRAIL
        # =============================================================

        print()
        print(
            "Checking protected risk constants..."
        )

        (
            guard_passed,
            guard_diffs,
        ) = guardrail_check(
            LIVE_DIR,
            work_dir,
        )

        if not guard_passed:

            print(
                "❌ PROTECTED RISK CONSTANTS CHANGED."
            )

            for name, (
                old,
                new,
            ) in guard_diffs.items():

                print(
                    f"{name}: {old} -> {new}"
                )

        else:

            print(
                "✅ Protected risk constants unchanged."
            )

        # =============================================================
        # AUDIT
        # =============================================================

        status = write_audit_report(
            work_dir=work_dir,
            guard_passed=guard_passed,
            guard_diffs=guard_diffs,
            test_passed=test_passed,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            commentary=commentary,
            bug_rounds=bug_rounds_completed,
            architecture_ran=architecture_ran,
        )

        print()
        print(
            f"VERIFIED_STATUS: {status}"
        )

        # =============================================================
        # ABSOLUTE MERGE GATE
        # =============================================================

        if status != "PASS":

            print(
                "❌ Automated merge gate FAILED."
            )

            print(
                "Candidate will NOT be merged."
            )

            commit_worktree(
                work_dir,
                f"Agent cycle #{cycle_count}: {status}",
            )

            return

        # =============================================================
        # COMMIT
        # =============================================================

        print()
        print(
            "Committing candidate..."
        )

        (
            commit_rc,
            commit_stdout,
            commit_stderr,
        ) = commit_worktree(
            work_dir,
            (
                f"Agent cycle #{cycle_count}: "
                f"verified PASS"
            ),
        )

        if commit_rc != 0:

            print(
                "❌ Candidate commit failed."
            )

            print(
                commit_stderr
            )

            return

        # =============================================================
        # GET CANDIDATE COMMIT
        # =============================================================

        rc, candidate_commit, stderr_commit = _run(
            [
                "git",
                "-C",
                work_dir,
                "rev-parse",
                "HEAD",
            ]
        )

        if rc != 0:

            print(
                "❌ Could not determine candidate commit."
            )

            print(
                stderr_commit
            )

            return

        candidate_commit = (
            candidate_commit.strip()
        )

        print(
            f"Candidate commit: "
            f"{candidate_commit}"
        )

        # =============================================================
        # FINAL PRE-MERGE SAFETY CHECK
        # =============================================================

        if live_trader_is_running():

            print()
            print(
                "⚠️ Live trader started before merge."
            )

            print(
                "Automatic merge CANCELLED."
            )

            print(
                f"Candidate preserved: {work_dir}"
            )

            return

        # =============================================================
        # AUTOMATIC MERGE + PUSH
        # =============================================================

        merge_success = (
            automatic_merge_and_push(
                branch_name,
                candidate_commit,
            )
        )

        if merge_success:

            print()
            print("=" * 70)
            print(
                "🎉 AUTOMATIC MERGE + PUSH COMPLETE"
            )
            print("=" * 70)

        else:

            print()
            print("=" * 70)
            print(
                "⚠️ AUTOMATIC MERGE/PUSH DID NOT COMPLETE"
            )
            print("=" * 70)

            print(
                f"Candidate branch preserved: "
                f"{branch_name}"
            )

        # =============================================================
        # DIFF
        # =============================================================

        diff = get_candidate_diff(
            LIVE_DIR,
            branch_name,
        )

        changed_lines = (
            len(
                diff.splitlines()
            )
            if diff
            else 0
        )

        print(
            f"Candidate diff size: "
            f"{changed_lines} lines"
        )

    except KeyboardInterrupt:

        print()
        print(
            f"🛑 Cycle #{cycle_count} interrupted."
        )

        if work_dir:

            print(
                f"Worktree preserved: "
                f"{work_dir}"
            )

        raise

    except Exception as exc:

        print()
        print("=" * 70)

        print(
            f"❌ ERROR IN CYCLE #{cycle_count}"
        )

        print("=" * 70)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        if work_dir:

            print(
                f"Worktree preserved: "
                f"{work_dir}"
            )

        print(
            "No automatic merge performed."
        )


# =====================================================================
# 22. MAIN
# =====================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)

    print(
        "🤖 AUTONOMOUS TRADING DEVELOPMENT TEAM"
    )

    print(
        "   GROQ + BUG-FIRST SAFE MODE"
    )

    print(
        "   AUTOMATIC MERGE + PUSH"
    )

    print("=" * 70)

    print(
        f"Live directory: "
        f"{LIVE_DIR}"
    )

    print(
        f"Worktree root: "
        f"{WORK_ROOT}"
    )

    print(
        f"Protected file: "
        f"{PROTECTED_FILE}"
    )

    print(
        f"Groq model: "
        f"{GROQ_MODEL}"
    )

    print(
        f"Git remote: "
        f"{REMOTE_NAME}"
    )

    try:

        print(
            f"Target branch: "
            f"{determine_target_branch()}"
        )

    except Exception as exc:

        print(
            f"Target branch: ERROR - {exc}"
        )

    print(
        f"Max bug rounds: "
        f"{MAX_BUG_FIX_ROUNDS}"
    )

    print(
        f"Max architecture rounds: "
        f"{MAX_ARCHITECTURE_ROUNDS}"
    )

    print(
        f"Cycle interval: "
        f"{CYCLE_SLEEP_SECONDS // 60} minutes"
    )

    print("=" * 70)

    cycle_count = 1

    try:

        while True:

            print()
            print("=" * 70)

            print(
                f"🔄 STARTING DEVELOPMENT CYCLE "
                f"#{cycle_count}"
            )

            print("=" * 70)

            run_cycle(
                cycle_count
            )

            print()
            print(
                f"💤 Sleeping "
                f"{CYCLE_SLEEP_SECONDS // 60} "
                f"minutes..."
            )

            print(
                "Press Ctrl+C to stop."
            )

            cycle_count += 1

            time.sleep(
                CYCLE_SLEEP_SECONDS
            )

    except KeyboardInterrupt:

        print()
        print(
            "🛑 Development team stopped."
        )