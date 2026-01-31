You are fixing code for a Google Colab environment.

## USER request
{user_prompt}

## Definition of Done (DOD) / KPIs (formal + testable)
The solution is DONE only if all KPIs below are met and verified by automated unit tests:

{kpis}

## Feedback from last run (must fix)
{feedback}

## Instructions
1) Produce TWO files:
   - solution.py : the implementation
   - test_solution.py : pytest unit tests
2) The tests MUST check that the KPIs are met (quantitatively and deterministically).
3) The solution must be runnable in Colab with only standard pip installs (avoid obscure system deps).
4) If assumptions are needed, encode them explicitly in code and in tests.
5) Keep the API stable: tests should import from solution.py.
6) Prefer small, reliable code; do not include try/catch or fallbacks.
7) Include docstrings and type hints.
8) There should be one function named main_notebook_call() that runs the entire task with no args.
   It will be tested to ensure it does not crash.

## Output format (STRICT)
Output exactly two fenced code blocks, in this order:

```python file=solution.py
<contents>
```

```python file=test_solution.py
<contents>
```

(No extra text outside those code blocks.)

Installed libraries and versions:
{pip_list}
