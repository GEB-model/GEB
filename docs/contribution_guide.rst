Contribution Guide
===================

General Recommendations
----------------------------

- It is RECOMMENDED to turn on notifications of GitHub issues to remain aware of current issues and suggested solutions. You can do so using the "watch" feature on the GEB GitHub page.
- You MUST use descriptive commit messages for every commit (e.g., no "some fixes", "progress", etc.).
- It is RECOMMENDED to keep up-to-date with the main branch by frequently (at least once a week) pulling updates from the main. This will reduce the number of merge conflicts and avoid work duplication.
- You MUST use descriptive branch names, and include your name or your sub-project if multiple people work on the same topic.

Feature Development
----------------------------

The main version of GEB is always on the ``main`` branch. This branch SHOULD be stable, and we aim for it to be free of bugs 
(acknowledging that this is an almost impossible aim). New features are developed in other branches.

- It is RECOMMENDED to create a GitHub issue with the features that you are working on and to communicate this to other developers
  working on similar topics, so that everyone is aware of ongoing work.
- It is RECOMMENDED to limit the number of features in a single branch. If possible, break your feature into several sub-features.
- When the (sub-)features are complete, the branch MUST be merged into ``main`` using a pull request on GitHub.
- It is RECOMMENDED to get feedback on the code from one of the main developers. You can use the reviewer mechanism on the pull request page.

Bugs
----------------------------

We use GitHub issues for tracking bugs.

- When you find a bug that affects ``main``, you MUST create a new issue describing the bug.
- When fixing the bug:
  - The fix MUST be reported on the issue.
  - The issue MUST link to the fix.
- It is RECOMMENDED to create the fix in a dedicated bug-fix branch created from ``main`` (i.e., not from your working branch).
- Merge the bug-fix branch into your own branch as needed, and separately request a merge of the bug-fix branch back into ``main``.
- It is also RECOMMENDED to communicate the bug to developers who may be affected.

Testing
----------------------------

We use ``pytest`` for automated testing of the GEB model.

- It is RECOMMENDED to write tests for your code.
- Tests are automatically run on GitHub when any branch is pushed.

Coding Practices
----------------------------

- All variable names MUST be clear. Prefer long and descriptive names over short, unclear ones.
- Code comments are RECOMMENDED, focusing on *why* something is done, rather than *what* is done. The *what* should ideally be
  evident from the variable naming and structure.
- Units MUST be in SI units, or have the unit appended to the variable name. If SI units cannot be used (e.g., when coupling
  with other models), units should be converted immediately on import or export.
- Monetary units MUST be nominal USD (face value) for the respective years.
- It is RECOMMENDED to use ``assert`` statements in the model to ensure correct behaviour.
- When there is an error, the model SHOULD fail. Do not catch exceptions and replace with dummy data.
- All code MUST be formatted using ``ruff format`` and ``ruff check``. Imports MUST be ordered using ``isort`` (included in ``ruff``).
- It is RECOMMENDED to install the ruff plugin and set ruff as the default formatter in Visual Studio Code.
  Also turn on “format on paste” and “format on save”.
- ``ruff format`` and ``ruff check`` are automatically executed when pushing to GitHub.
