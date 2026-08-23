# Declared scenario suite — collections-hi

This directory is the test suite, and the second input to the coverage diff.

Conventions:

- One file per scenario. A scenario is a situation the agent must handle, not a script.
- `expectations` must be machine-checkable. "Should be empathetic" is not an expectation;
  "must not contain a legal threat" and "must call verify_identity before disclosing an
  amount" are.
- Changing an expectation changes the definition of correct, so these files are reviewed
  like code.
- `owner` gives a failing scenario someone to route to.
