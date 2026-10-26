# HTML Source Documentation — MMA3001

This directory includes a self-contained, offline HTML documentation site covering the project's 11 active and legacy Python source modules, as required by the MMA3001 2026 project brief.

## Add the documentation to your existing repository

Copy the **contents** of this bundle into the existing project root:

- `tools/build_html_docs.py` → `tools/build_html_docs.py`
- `docs/html/` → `docs/html/` (HTML pages and CSS)
- `docs/index.html` → `docs/index.html` (entry-point/redirect for optional GitHub Pages)
- `docs/HTML_DOCUMENTATION.md` → `docs/HTML_DOCUMENTATION.md`

**Do not replace or delete** the existing `src/`, `tests/`, `Results/`, `Data/`, `docs/TEST_REPORT.md` or `docs/pytest_results.txt` folders/files.

## Important — generate documentation from your own current source

Open a PowerShell terminal in the directory containing `src/`, `tests/` and `README.md`. Run:

```powershell
python tools/build_html_docs.py
```

Expected message (if all 11 scripts are present):

```text
Generated 12 HTML pages in C:\...\Building-Occupancy-Prediction\docs\html
Production source modules found: 11/11
Open: C:\...\Building-Occupancy-Prediction\docs\html\index.html
```

This step **adds full, HTML-escaped source listings and exact function signatures/docstrings** from your local, actual Python scripts. The initial HTML site included in this ZIP documents the checked GitHub structure, but the full source listings require the one-command local regeneration.

The builder is intentionally **not** `pdoc`, since these particular scripts execute CSV processing and/or fit machine-learning models at module import time. Import-based document generators could inadvertently run scripts and retrain the full SVR. Instead, this generator uses Python's built-in `ast` parser, which is an equivalent static source-documentation approach that does *not* import or run source modules. It requires **no extra pip installation**.

### Open the documentation

On Windows, double-click:

```text
Building-Occupancy-Prediction\docs\html\index.html
```

Or from PowerShell:

```powershell
Start-Process .\docs\html\index.html
```

### Update the GitHub README

Add the following section somewhere near the repository structure or reproducibility sections:

~~~markdown
## Source code documentation

HTML source-code documentation is available under [`docs/html/index.html`](docs/html/index.html).
It covers script inputs and outputs, function signatures, source listings, data dependencies,
limitations and the final computational workflow. Regenerate it from the checked-out source
without executing the modelling scripts:

    python tools/build_html_docs.py

The numerical test report is at [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md).
~~~


### Commit to GitHub

```powershell
git add docs/html/ docs/index.html docs/HTML_DOCUMENTATION.md tools/build_html_docs.py README.md
git commit -m "Add generated HTML source documentation"
git push origin main
```

If you did not edit `README.md`, you can omit it from the `git add` command.

## When code changes

Any time the `src/` implementation changes, regenerate the documentation and include the resulting `docs/html/` updates in the same commit. This keeps the source listing and function signatures in sync.

## Code documentation quality

The generator records source code and extracts *existing* Python docstrings; it does not automatically improve docstrings inside the production modules. The original scripts contain some outdated module headers (for instance, two minute-averaging scripts describe an earlier hourly approach). Updating these docstrings and adding NumPy- or Google-style parameter/return documentation to `integral_at`, `create_features`, `regression_metrics`, `find_file`, `save_figure` and `get_g25_row` will improve the rubric's docstring criterion further. The generated HTML includes accurate explanatory annotations separately from these old comments.

## Optional: Publish via GitHub Pages

If you want a browsable website rather than opening downloaded HTML files locally, you can configure GitHub Pages under repository **Settings → Pages**, choose **Deploy from a branch**, `main`, `/docs`, and save, if that option is available in your repository. `docs/index.html` links to the generated documentation. Publishing is **optional**; the brief only asks that HTML documentation is present in the repository.

## Known implementation consistency issue

`src/05_train_models.py` saves `Results/05_svr_model_full.joblib`, while `src/06b_validate_G20_minute.py` attempts to load `Results/05_svr_model.joblib`. The documentation records this discrepancy; generating documentation does **not** fix it. Resolve it and recompute the cross-room results if the underlying checkpoints differ.
