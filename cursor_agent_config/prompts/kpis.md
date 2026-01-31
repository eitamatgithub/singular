You are an expert software engineer.

Given the following user prompt, generate a list of KPIs (as free text) that can be
used to verify the solution meets the user's requirements.

User prompt:
{user_prompt}

Requirements:
- Every error (unresolved or runtime) should fail tests; avoid try/catch/fallbacks.
- Tests must check behavior and the presence of a main_notebook_call() entry point.
- Keep KPIs precise and testable.

Keep in mind these libraries are currently installed:
{pip_list}
