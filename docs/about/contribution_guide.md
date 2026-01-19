# Contribution Guide

## Developing with git

In this guide, we will assume you use VS Code and already set up GEB for [development](../getting_started/installation.md) and have a basic knowledge of git ([here](https://www.codecademy.com/learn/learn-git) is a nice course).

1. [create a new branch](https://code.visualstudio.com/docs/sourcecontrol/branches-worktrees), which you name `your_name-feature_name`. For example, when Alice wants to allow people in the model to have pets, the branch could be called `alice-pets`. This allows you to develop the feature independently from the main branch.
2. Develop your feature or do the bugfix. Try to keep this to a single feature or bugfix.
3. After you are done, stage and commit your code. Make sure to write a clear commit message. More info on staging and commiting can be found [here](https://code.visualstudio.com/docs/sourcecontrol/staging-commits). If you are working on something larger, there may also be multiple commits.
4. After you finished the feature or bugfix, make sure to push all your changes to GitHub, where you should be able to find the branch with your changes.
5. Now you can [open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request). This is a request to merge your code with the main branch.
6. Once you opened the request, there will first be an automated review from copilot after a few minutes. Have a look at these suggestions and implement them if you feel they are useful. Note that this is AI so not everything is useful, but it often allows you to catch some errors or make some clarifications.
7. In addition, all sorts of other automated checks are performed related to formatting and tests. Ensure that all tests tests pass.
8. Then, [request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review) someone (a human this time) to review your code. They will review your code and approve the code or request you to make changes. If they request changes, make these changes and request a re-review.
9. Once the code is approved, you or the reviewer can merge the branch with the main branch and delete the feature branch (so we keep things clean).
10. Congrats your code is now in the main branch!

## Contributing to documentation

First of all, thank you for (thinking about) contributing to the documentation! When you are a member of the GEB repository, you can edit documentation yourself. Pull requests that only edit the documentation are automatically approved. For outside contributors, please make a pull request and we will merge it for you.

You can also test your changes locally first. For example, this command will build the documentation and create a local server so that you can view the documentation in your browser.

```sh
uv run mkdocs serve
```

It is also possible to just build the documentation locally:

```sh
uv run mkdocs build
```

Your files will end up in the folder `site`. Open `index.html` in your browser for the main page.

## Making a release

To create a release of GEB, and creating a package on PyPi, take the following steps.

1. Bump the version of GEB using `uv version --bump xxxx`. Replacing xxxx with major, minor, patch, stable, alpha, beta, rc, post or dev.
2. Update the [changelog](https://github.com/GEB-model/GEB/blob/main/CHANGELOG.md), moving the updates to the appropriate version (keeping the dev header for the next version).
3. Create a [new release](https://github.com/GEB-model/GEB/releases). Both the tag and the title should match the new version number. In the release notes, paste the relevant updates from the [changelog](https://github.com/GEB-model/GEB/blob/main/CHANGELOG.md).
4. Now, a [GitHub action](https://github.com/GEB-model/GEB/actions/workflows/publish.yml) should automatically start and upload the new version to PyPi. Confirm the release is indeed available [here](https://pypi.org/project/geb/). 

## Rules and recommendations

### General Recommendations

- It is RECOMMENDED to turn on notifications of GitHub issues to remain aware of current issues and suggested solutions. You can do so using the "watch" feature on the GEB GitHub page.
- You MUST use descriptive commit messages for every commit (e.g., no "some fixes", "progress", etc.).
- It is RECOMMENDED to keep up-to-date with the main branch by frequently (at least once a week) pulling updates from the main. This will reduce the number of merge conflicts and avoid work duplication.
- You MUST use your name followed by a descriptive branch name. For example, `tim-loves-forecasting`.

### Feature Development

The main version of GEB is always on the `main` branch. This branch SHOULD be stable, and we aim for it to be free of bugs (acknowledging that this is an almost impossible aim). New features are developed in other branches.

- It is RECOMMENDED to create a GitHub issue with the features that you are working on and to communicate this to other developers working on similar topics, so that everyone is aware of ongoing work.
- It is RECOMMENDED to limit the number of features in a single branch. If possible, break your feature into several sub-features.
- When the (sub-)features are complete, the branch MUST be merged into `main` using a pull request on GitHub.
- It is REQUIRED to get feedback on the code from one of the main developers. You can use the reviewer mechanism on the pull request page.

### Bugs

We use GitHub issues for tracking bugs.

- When you find a bug that affects `main`, you MUST create a new issue describing the bug.
- When fixing the bug:
  - The fix MUST be reported on the issue page.
  - The issue MUST link to the fix.
- It is RECOMMENDED to create the fix in a dedicated bug-fix branch created from `main` (i.e., not from your working branch). Then, merge the bug-fix branch into your own branch as needed, and separately request a merge of the bug-fix branch back into `main`.
- Alternatively, you can make a fix directly in your working branch and create a commit that fixes the bug, and only the bug. Then, use git cherry-pick to apply the fix to the `main` branch.
- It is also RECOMMENDED to communicate the bug to developers who may be affected.

### Testing

We use `pytest` for automated testing of the GEB model.

- It is RECOMMENDED to write tests for your code.
- Unit tests are automatically run on GitHub when any branch is pushed.
- Integration tests are RECOMMENDED to run locally before pushing to GitHub. They can be run using the command `pytest` or in vscode.

### Coding Practices

- All variable names MUST be clear. Prefer long and descriptive names over short, unclear ones.
- Function documentation MUST be provided for all functions, including the function's purpose, parameters, and return values. Use the Google style for docstrings.
- Type annotations MUST be used for all functions and methods, both for arguments and return values.
- Code comments are RECOMMENDED, focusing on *why* something is done, rather than *what* is done. The *what* should ideally be evident from the variable naming and structure.
- Units MUST be in SI units, or have the unit appended to the variable name. If SI units cannot be used (e.g., when coupling with other models), units should be converted immediately on import or export.
- Monetary units MUST be nominal USD (face value) for the respective years.
- It is RECOMMENDED to use `assert` statements in the model to ensure correct behaviour.
- When there is an error, the model SHOULD fail. Do not catch exceptions and replace with dummy data.
- All code MUST be formatted using `ruff format` and `ruff check`. Imports MUST be ordered using `isort` (included in `ruff`). It is RECOMMENDED to install the ruff plugin and set ruff as the default formatter in Visual Studio Code. Also turn on "format on paste" and "format on save". `ruff format` and `ruff check` are automatically executed when pushing to GitHub.
