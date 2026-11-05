---
title: Contribution Guide
summary: Contributing to the project
---
# Contribution Guide
This guide provides information on building and contributing to the `giftwrap` project.

## Required Tooling
This project is managed using the [uv](https://docs.astral.sh/uv/) tooling system, a modern dependency manager for python projects. Please ensure that it is available in your environment.

## Installing Dependencies
To install the project dependencies, run the following command:

```bash
uv sync --all-extras --all-groups
```

Note that when adding/updating packages with `uv add` or updating the `pyproject.toml` file, you should run `uv sync` to ensure that all dependencies are correctly installed and the `uv.lock` file is updated.

## Building the package
To build the package, use the following command:

```bash
uv build
```

You should see a build `.whl` file in the `dist/` directory. This can be directly installed with `pip`. 


## Running the Tests
The test suite lives in `tests/` and uses [pytest](https://docs.pytest.org/). It needs every optional extra and the `dev` dependency group, so install with `uv sync --all-extras --all-groups` first. Then run:

```bash
uv run pytest
```

A healthy run reports only `passed` and `xfailed` results; any `failed`, `error`, or `XPASS` result needs attention. The full suite takes about two minutes and uses no network access.

Useful variations:

| Command | Purpose |
|---|---|
| `uv run pytest tests/test_step4_collect_counts.py` | Run the tests for a single module. |
| `uv run pytest -m "not slow and not multiprocess"` | Skip tests that load large bundled resources or spawn worker processes. |
| `uv run pytest -m integration` | Run only the end-to-end pipeline tests. |
| `uv run pytest -m known_defect -rx` | List the tests that document confirmed, not-yet-fixed bugs. |
| `uv run pytest --cov=giftwrap --cov-branch --cov-report=term-missing` | Measure line and branch coverage. |

### Known defects
Bugs that have been confirmed but not yet fixed are captured as tests marked `known_defect`. Each test asserts the *intended* behavior and is marked as a strict expected failure (`xfail`), so the suite stays green while the bug exists. When you fix one of these bugs, its test will start passing and pytest will report it as `XPASS(strict)`, which fails the run. Remove that test's `known_defect` and `xfail` markers in the same change, so the test becomes a regular regression test.

See `tests/README.md` for the layout of the suite, the shared fixtures, and conventions for writing new tests.


## Updating Documentation
The `giftwrap` documentation site is built using [mkdocs](https://www.mkdocs.org/) and [mkdocs-material](https://squidfunk.github.io/mkdocs-material/). To update the documentation, follow these steps:

1. Make your changes to the Markdown files in the `docs/` directory.

2. If adding a new page, ensure it is linked in the `nav` section of the `mkdocs.yml` file.

3. To preview your changes locally, run:
   ```bash
   uv run mkdocs serve
   ```
   This will start a local server where you can view the documentation at `http://localhost:8000`.

4. When committed to Github, the documentation will automatically be build and deployed.


## Releasing a new version (Maintainers only)
To release a new version of the package, follow these steps:

1. Update the version in `pyproject.toml` to the new version.

2. Run the build command to create the new package:
   ```bash
   uv build
   ```

3. Upload the new package to PyPI via uv or through Github Actions (Recommended):
    ```bash
    # Manually via command line
    uv publish --token <your-pypi-token>
    ```
    or simply create a new release on GitHub with a tag in the form of `vX.Y.Z`, and the GitHub Actions workflow will automatically publish the package to PyPI.
