#!/usr/bin/env python3
"""Build offline HTML documentation for the MMA3001 occupancy prediction project.

This tool deliberately uses Python's AST parser rather than importing project
modules. Several source scripts perform large CSV reads, model fitting, or plot
creation at import time, so importing them during documentation generation
would be unsafe and unnecessarily expensive.

Usage (run from the project root):
    python tools/build_html_docs.py

Dependencies: Python >=3.9 standard library only.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import re
from typing import Any

REPO_URL = "https://github.com/jsau0014-gif/MMA3001-Building-occupancy-prediction-regression-methods"
SEVEN_FEATURES = ["TimeSin", "TimeCos", "DaySin", "DayCos", "MonthSin", "MonthCos", "IsWeekend"]

@dataclass(frozen=True)
class ModuleInfo:
    path: str
    title: str
    role: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    method: str
    cautions: tuple[str, ...] = ()
    primary: bool = True


MODULES = [
    ModuleInfo(
        "src/01_Extract_Occupancy_Data.py", "01 · Extract room records", "Data extraction",
        ("Data/5occupancySensor_MayToDec2024_9MRows.csv",),
        ("Data/processed/G25_Occupancy.csv", "Data/processed/G20_Occupancy.csv"),
        "Reads the source occupancy CSV in 100,000-row chunks, selects floorspace IDs for G.25 and G.20, and appends the matching sensor records to room-specific CSV files.",
        ("Requires the original large occupancy dataset and Git LFS materialisation.",)
    ),
    ModuleInfo(
        "src/02_Inspect_Occupancy_Data.py", "02 · Inspect extracted measurements", "Input quality review",
        ("Data/processed/G25_Occupancy.csv", "Data/processed/G20_Occupancy.csv"),
        ("Console summaries of dates, headcounts, occupancy status and missing values",),
        "Parses timestamps as UTC and converts to Australia/Melbourne before reporting the coverage and distributions of each room. Does not train a model."
    ),
    ModuleInfo(
        "src/03_preprocess_training.py", "03 · Time-weighted G.25 averages", "Numerical preprocessing",
        ("Data/processed/G25_Occupancy.csv",),
        ("Data/processed/G25_Minute_Training.csv",),
        "Sorts valid time-stamped measurements, resolves repeated timestamps by retaining the final value, models headcount as a zero-order-hold function and computes one-minute time-weighted averages using a cumulative occupancy–time integral.",
        ("Only minutes containing at least one original valid reading are retained.",
         "The first partially covered minute is divided by the observed duration rather than a full 60 seconds.",
         "The legacy top-level script docstring mentions hourly sampling, but the implementation produces minute averages.")
    ),
    ModuleInfo(
        "src/03b_preprocess_G20.py", "03b · Time-weighted G.20 averages", "Numerical preprocessing",
        ("Data/processed/G20_Occupancy.csv",),
        ("Data/processed/G20_Minute_validation.csv",),
        "Applies the same cumulative zero-order-hold occupancy–time integration to G.20, producing like-for-like one-minute averages for independent cross-room validation.",
        ("Minutes without original readings are omitted; long measurement gaps remain a modelling limitation.",
         "The legacy top-level script docstring mentions hourly sampling, but the implementation produces minute averages.")
    ),
    ModuleInfo(
        "src/04_feature_engineering.py", "04 · Engineer G.25 model inputs", "Feature engineering",
        ("Data/processed/G25_Minute_Training.csv", "Data/processed/G20_Occupancy.csv"),
        ("Data/processed/G25_Model_Train.csv", "Data/processed/G20_Model_Validation.csv (older raw-target path)"),
        "Converts timestamps to Melbourne local time and computes seven cyclical calendar-derived predictors. Also writes a raw G.20 features file retained for an earlier comparison, not the final minute-averaged cross-room experiment.",
        ("Use 04b_feature_engineering_G20.py to generate the final G.20 validation features.",)
    ),
    ModuleInfo(
        "src/04b_feature_engineering_G20.py", "04b · Engineer G.20 validation inputs", "Feature engineering",
        ("Data/processed/G20_Minute_validation.csv",),
        ("Data/processed/G20_Model_Validation_Minute.csv",),
        "Applies the same seven temporal features to the minute-averaged G.20 observations for independent room evaluation."
    ),
    ModuleInfo(
        "src/05_train_models.py", "05 · Train and compare regressors", "Model development",
        ("Data/processed/G25_Model_Train.csv",),
        ("Results/05_model_comparison_random.csv", "Results/05_linear_model.joblib", "Results/05_decision_tree.joblib", "Results/05_svr_model_full.joblib", "Results/05_*.png and CSV evaluation artefacts"),
        "Uses a shuffled, reproducible 80/20 split of G.25 (random_state=42). Fits Multiple Linear Regression, a tuned Decision Tree and a scaled RBF Support Vector Regressor. Computes MAE, RMSE and R² against a training-mean baseline; records tuning sensitivity and plots.",
        ("The full 298,608-row RBF SVR training may take several hours.",
         "An old header comment refers to a chronological split; the actual implementation uses train_test_split with shuffle=True.",
         "The saved full SVR checkpoint is named 05_svr_model_full.joblib.")
    ),
    ModuleInfo(
        "src/06b_validate_G20_minute.py", "06b · Independent G.20 validation", "Model evaluation",
        ("Data/processed/G20_Model_Validation_Minute.csv", "Results/05_linear_model.joblib", "Results/05_decision_tree.joblib", "Results/05_svr_model.joblib (current lookup)"),
        ("Results/06b_G20model_comparison.csv", "Results/06b_G20validation_predictions.csv", "Results/06b_G20*.png plots"),
        "Loads fitted G.25 models without retraining. Evaluates the processed G.20 observations and reports MAE, RMSE, R², mean signed residual (actual − predicted), runtime and diagnostic plots.",
        ("Important checkpoint mismatch: this script looks for 05_svr_model.joblib, while script 05 writes 05_svr_model_full.joblib. Align and re-evaluate before making a cross-room full-model claim.",
         "G20_SAMPLE_SIZE acts as a cap; a cap greater than available observations means all rows are evaluated.")
    ),
    ModuleInfo(
        "src/07_generate_report_figures.py", "07 · Generate report figures", "Reporting",
        ("Results/05_model_comparison_random.csv", "Results/06b_G20model_comparison.csv or Results/06b_G20_model_comparison.csv", "Results/06b_G20validation_predictions.csv or Results/06b_G20_validation_predictions.csv", "Additional optional result CSVs"),
        ("Results/report_figures/*.png",),
        "Produces final comparison, error, occupancy-distribution and computational-cost figures from previously saved metrics. Does not fit models or generate new predictions.",
        ("Optional figures may be skipped when earlier experimental comparison CSVs are absent.",)
    ),
    ModuleInfo(
        "src/03_Preprocess_hourly.py", "Legacy · Hourly preprocessing", "Legacy workflow",
        ("Data/processed/G25_Occupancy.csv", "Data/processed/G20_Occupancy.csv"),
        ("Data/processed/G25_Hourly.csv", "Data/processed/G20_Hourly.csv"),
        "Earlier hourly-resolution experiment. It remains in src/ for traceability but is not used to reproduce the final one-minute-averaged experiment.", primary=False
    ),
    ModuleInfo(
        "src/06_validate_G20.py", "Legacy · Raw G.20 validation", "Legacy workflow",
        ("Data/processed/G20_Model_Validation.csv", "Previously saved G.25 model checkpoints"),
        ("Results/06_G20*.csv/png",),
        "Earlier cross-room validation on raw G.20 readings. Kept for traceability; use 06b_validate_G20_minute.py for the final like-for-like minute-level evaluation.", primary=False
    ),
]

CURATED_FUNCTION_DOCS = {
    "integral_at": (
        "Evaluate the cumulative zero-order-hold occupancy–time integral at the given timestamp boundaries.",
        "boundaries_ns: array-like integer nanosecond timestamp boundaries",
        "Array of integrated occupancy × time at each requested boundary; used to form minute-average differences.",
    ),
    "create_features": (
        "Append seven timestamp-derived features to a DataFrame, preserving the target and timestamp for later filtering.",
        "data: pandas.DataFrame containing a parseable Timestamp column",
        "pandas.DataFrame with TimeSin, TimeCos, DaySin, DayCos, MonthSin, MonthCos and IsWeekend.",
    ),
    "regression_metrics": (
        "Evaluate regression prediction accuracy using an absolute-error, squared-error and variance-explained summary.",
        "y_true: observed headcounts; y_pred: predicted headcounts",
        "Dictionary with MAE, RMSE and R2.",
    ),
    "find_file": (
        "Return the first existing filename within Results/ from several supported legacy naming variants.",
        "*filenames: candidate filenames in priority order",
        "pathlib.Path of the first existing file; raises FileNotFoundError when none exists.",
    ),
    "save_figure": (
        "Save the current Matplotlib figure to Results/report_figures, with a fallback to Results/.",
        "filename: output PNG filename",
        "No value returned; current figure is closed.",
    ),
    "get_g25_row": (
        "Select the appropriate G.25 comparison-metrics row by its model-name prefix.",
        "prefix: model label prefix for the metrics DataFrame index",
        "Selected row of internal-validation metrics.",
    ),
}

WORKFLOW = [
    ("01 · Extract", "src/01_Extract_Occupancy_Data.py", "Room CSVs"),
    ("02 · Inspect", "src/02_Inspect_Occupancy_Data.py", "Data checks"),
    ("03 / 03b · Average", "src/03_preprocess_training.py", "One-minute mean targets"),
    ("04 / 04b · Features", "src/04_feature_engineering.py", "Seven time inputs"),
    ("05 · Train", "src/05_train_models.py", "G.25 models + scores"),
    ("06b · Validate", "src/06b_validate_G20_minute.py", "G.20 transfer scores"),
    ("07 · Figures", "src/07_generate_report_figures.py", "Report plots"),
]

CSS = r"""
:root {--navy:#132138;--ink:#192637;--muted:#5b6d7d;--line:#dce3ec;--bg:#f6f8fb;--card:#fff;--blue:#285fa1;--pale:#eaf1fc;--orange:#956212;--orange-bg:#fff7e9;--teal:#146b60;--teal-bg:#ecf8f5}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.62 -apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif}
a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}
.shell{display:grid;grid-template-columns:275px minmax(0,1fr);min-height:100vh}.side{padding:30px 21px;background:var(--navy);color:#d9e4f2;position:sticky;top:0;height:100vh;overflow:auto}
.brand{color:#fff;font-weight:800;font-size:17px;line-height:1.4}.brand small{display:block;font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#a6bad2;margin-bottom:7px}
.side .subhead{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#99b0cc;margin:27px 9px 7px}
.side a{display:block;color:#ccdcf0;font-size:12.5px;padding:7px 10px;border-radius:7px;overflow-wrap:anywhere}.side a:hover,.side a.active{background:#2a4160;color:#fff;text-decoration:none}[hidden]{display:none!important}
.main{padding:45px clamp(20px,5.4vw,85px) 90px;min-width:0;max-width:1390px}
.breadcrumb{font-size:12px;color:var(--muted);margin-bottom:19px}.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.14em;color:var(--blue);font-weight:800}
h1{line-height:1.18;letter-spacing:-.036em;font-size:clamp(30px,3vw,43px);margin:10px 0 17px}h2{font-size:22px;margin:38px 0 16px;letter-spacing:-.025em}h3{font-size:16px;margin:22px 0 10px}
p{margin:9px 0 17px;max-width:90ch}.lead{font-size:16px;color:#42556a;max-width:78ch}
.card{border:1px solid var(--line);border-radius:12px;background:var(--card);padding:20px 22px;margin:15px 0}.card h3{margin-top:0}.tag{display:inline-block;border:1px solid #bad2eb;background:#edf5ff;color:#1e567f;font-size:11px;letter-spacing:.02em;font-weight:700;padding:3px 8px;border-radius:6px;margin-right:8px}
.badge-legacy{background:#f1f1f1;color:#676c73;border-color:#ddd}.muted{color:var(--muted)}.small{font-size:12px}.hint{background:var(--teal-bg);border-left:3px solid var(--teal);padding:14px 16px;border-radius:0 8px 8px 0;margin:22px 0}
.warn{background:var(--orange-bg);border-left:3px solid #d68c18;padding:14px 16px;border-radius:0 8px 8px 0;margin:22px 0}.warn strong{color:var(--orange)}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}.cards .card{margin:0}.cards h3{font-size:15px;line-height:1.35}.cards p{font-size:13px;color:var(--muted)}
ul{padding-left:20px}li{margin:6px 0}code,pre{font:12px/1.6 Consolas,'SFMono-Regular',monospace}code{background:#eaf0f6;padding:2px 4px;border-radius:4px;overflow-wrap:anywhere}pre{padding:16px 18px;border-radius:8px;background:#16263c;color:#edf4fb;overflow:auto;max-height:680px;tab-size:4}pre code{padding:0;background:transparent;color:inherit}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}table{width:100%;border-collapse:collapse;font-size:13px;background:#fff}th,td{text-align:left;border-bottom:1px solid var(--line);padding:12px 14px;vertical-align:top}th{background:#edf2f8;color:#304963;font-size:11px;letter-spacing:.03em;text-transform:uppercase}tr:last-child td{border:0}
.path-list{list-style:none;padding:0}.path-list li{padding:7px 0;border-bottom:1px solid var(--line);font-size:13px}.path-list li:last-child{border:0}
.workflow{display:flex;gap:9px;flex-wrap:wrap;margin:18px 0}.step{background:#fff;border:1px solid var(--line);padding:10px 12px;border-radius:9px;min-width:140px;flex:1}.step strong{display:block;font-size:12px}.step span{display:block;font-size:11px;color:var(--muted)}
.search{width:100%;padding:13px 16px;background:#fff;border:1px solid #cad7e6;border-radius:9px;font-size:14px;margin:0 0 22px;outline-offset:2px}
.function{padding:16px 0 10px;border-bottom:1px solid var(--line)}.function:last-child{border:0}.function h3{font-family:Consolas,monospace;font-size:14px;overflow-wrap:anywhere}
details{margin:19px 0;border:1px solid var(--line);border-radius:9px;background:white;padding:12px 16px}summary{cursor:pointer;font-weight:700}footer{border-top:1px solid var(--line);margin-top:45px;padding-top:18px;font-size:12px;color:var(--muted)}
@media(max-width:850px){.shell{display:block}.side{position:relative;height:auto;padding:16px}.side .subhead{display:none}.side-links{display:none}.main{padding:22px 18px 50px}h1{font-size:30px}}@media print{.side,.search{display:none}.shell{display:block}.main{padding:0}.card{break-inside:avoid}details{break-inside:avoid}}
"""


def e(s: Any) -> str:
    return escape(str(s), quote=True)


def code(s: Any) -> str:
    return f"<code>{e(s)}</code>"


def section_list(items: tuple[str,...] | list[str]) -> str:
    return '<ul class="path-list">' + ''.join(f'<li>{code(x)}</li>' for x in items) + '</ul>'


def slug(path: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]+', '_', path.replace('.py', '')).strip('_') + '.html'


def source_link(path: str) -> str:
    return f"{REPO_URL}/blob/main/{path}"


def split_modules() -> tuple[list[ModuleInfo], list[ModuleInfo]]:
    return ([m for m in MODULES if m.primary], [m for m in MODULES if not m.primary])


def nav(current: str, prefix: str) -> str:
    main, old = split_modules()
    links = [('<a href="'+prefix+'index.html" class="'+('active' if current=='index' else '')+'">Overview &amp; workflow</a>')]
    links += ['<div class="subhead">Main source modules</div>']
    for m in main:
        href = prefix+'modules/'+slug(m.path)
        active = 'active' if current == m.path else ''
        links.append(f'<a class="{active}" href="{e(href)}">{e(m.path.split("/")[-1])}</a>')
    links += ['<div class="subhead">Earlier implementations</div>']
    for m in old:
        links.append(f'<a href="{e(prefix+"modules/"+slug(m.path))}" class="'+('active' if current==m.path else '')+'">'+e(m.path.split('/')[-1])+'</a>')
    links += ['<div class="subhead">Other resources</div>',
              f'<a href="{REPO_URL}/tree/main/tests">Pytest source ↗</a>',
              f'<a href="{REPO_URL}/blob/main/docs/TEST_REPORT.md">Test report ↗</a>',
              f'<a href="{REPO_URL}/blob/main/README.md">README &amp; setup ↗</a>']
    return ''.join(links)


def layout(title: str, current: str, content: str, prefix: str = '') -> str:
    css_href = prefix + 'assets/style.css'
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)} | MMA3001 Source Documentation</title>
<meta name="description" content="Numerical methods and machine learning source documentation for Building Occupancy Prediction at Monash University.">
<link rel="stylesheet" href="{e(css_href)}"></head><body>
<div class="shell"><aside class="side"><div class="brand"><small>MMA3001 · 2026</small>Building Occupancy<br>Source Documentation</div>
<div class="side-links">{nav(current,prefix)}</div></aside><main class="main">{content}
<footer>Generated by <code>tools/build_html_docs.py</code>. Source links refer to the GitHub <code>main</code> branch. Code is documented by parsing Python source; modules are not imported or executed.</footer></main></div>
</body></html>'''


def get_functions(root: Path, m: ModuleInfo) -> tuple[list[dict[str,str]], str|None, str|None, int|None]:
    """Statically read the source, signatures and docstrings if locally present."""
    f = root / m.path
    if not f.is_file():
        return [], None, None, None
    source = f.read_text(encoding='utf-8-sig')
    try:
        tree = ast.parse(source, filename=str(f))
    except SyntaxError as exc:
        return [], source, f"AST parse error at line {exc.lineno}: {exc.msg}", len(source.splitlines())
    funcs = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # AST gives a reliable signature even when annotations contain colons
            # or the definition spans multiple lines.
            params = ast.unparse(n.args)
            returns = f' -> {ast.unparse(n.returns)}' if n.returns is not None else ''
            keyword = 'async def' if isinstance(n, ast.AsyncFunctionDef) else 'def'
            signature = f"{keyword} {n.name}({params}){returns}:"
            funcs.append({
                'name': n.name,
                'signature': signature,
                'line': str(n.lineno),
                'docstring': ast.get_docstring(n) or '',
            })
    return funcs, source, None, len(source.splitlines())


def render_function(name: str, signature: str, line: str | None, docstring: str | None = None) -> str:
    known = CURATED_FUNCTION_DOCS.get(name)
    explanation = known[0] if known else ('Original code docstring: ' + docstring if docstring else 'See linked source for algorithm details.')
    signature = signature or ('def '+name+'(...):')
    line_note = f'<span class="small muted">Defined at source line {line}</span>' if line else '<span class="small muted">See source file for definition</span>'
    inp = f'<p><strong>Input:</strong> {e(known[1])}</p>' if known else ''
    out = f'<p><strong>Output:</strong> {e(known[2])}</p>' if known else ''
    doc_text = f'<details><summary>Original Python docstring</summary><pre><code>{e(docstring)}</code></pre></details>' if docstring else ''
    return f'<section class="function"><h3>{e(signature)}</h3>{line_note}<p>{e(explanation)}</p>{inp}{out}{doc_text}</section>'


def render_module(m: ModuleInfo, root: Path) -> str:
    funcs, source, parsing_err, line_count = get_functions(root, m)
    status = 'Legacy / not used by final workflow' if not m.primary else m.role
    badge_cls = 'badge-legacy' if not m.primary else ''
    src_label = f'{line_count:,} lines indexed locally' if line_count is not None else 'Source browse link available'
    content = f'''
    <div class="breadcrumb"><a href="../index.html">Overview</a> / {e(m.path)}</div>
    <div class="eyebrow">Source module · {e(status)}</div>
    <h1>{e(m.title)}</h1>
    <p class="lead">{e(m.method)}</p>
    <div><span class="tag {badge_cls}">{e(m.role)}</span> <span class="small muted">{e(src_label)}</span> ·
    <a target="_blank" rel="noopener" href="{e(source_link(m.path))}">View current source on GitHub ↗</a></div>
    <div class="cards">
    <section class="card"><h3>Required inputs</h3>{section_list(m.inputs)}</section>
    <section class="card"><h3>Files and results produced</h3>{section_list(m.outputs)}</section>
    </div>'''
    if m.cautions:
        content += '<div class="warn"><strong>Usage notes and limitations</strong><ul>' + ''.join(f'<li>{e(s)}</li>' for s in m.cautions) + '</ul></div>'
    if parsing_err:
        content += f'<div class="warn"><strong>Parsing warning:</strong> {e(parsing_err)}</div>'
    content += '<h2>Function reference</h2>'
    if funcs:
        content += '<p class="muted small">Signatures and line numbers are extracted from the local production source without executing it.</p>'
        content += '<div class="card">'+ ''.join(render_function(f['name'],f['signature'],f['line'],f['docstring']) for f in funcs) +'</div>'
    else:
        key_func_names = {'03_preprocess_training.py':['integral_at'], '03b_preprocess_G20.py':['integral_at'],
                         '04_feature_engineering.py':['create_features'],'04b_feature_engineering_G20.py':['create_features'],
                         '05_train_models.py':['regression_metrics'], '07_generate_report_figures.py':['find_file','save_figure','get_g25_row']}
        names = key_func_names.get(m.path.split('/')[-1],[])
        if names:
            content += '<p class="small muted">Known function contracts are listed below. Rebuild locally to extract exact signatures, source lines and docstrings.</p><div class="card">' + ''.join(render_function(name,'',None) for name in names)+'</div>'
        else:
            content += '<p>This module implements its principal operations at top level. Executing the script triggers its workflow; it is not a callable API module. See the input/output contract above and the GitHub source.</p>'
    content += '<h2>Source listing</h2>'
    if source is not None:
        content += '<p class="muted small">The source below is a static copy from the local working tree at the time documentation was built; refresh after code changes.</p>'
        content += '<details><summary>Expand full source code (HTML escaped)</summary><pre><code>'+e(source)+'</code></pre></details>'
    else:
        content += ('<div class="hint">The HTML pages in this ZIP are ready to browse. To additionally embed the exact source listing and discover every function signature, '
                    'copy <code>tools/build_html_docs.py</code> into the project root and run <code>python tools/build_html_docs.py</code> on your own clone.</div>')
    return layout(m.title,m.path,content,prefix='../')


def index(root: Path) -> str:
    prim, legacy = split_modules()
    steps = ''.join(f'<div class="step"><strong>{e(label)}</strong><span>{e(out)}</span></div>' for label,path,out in WORKFLOW)
    cards = ''.join(f'<article class="card" data-search="{e((m.path+" "+m.role+" "+m.method).lower())}">'
                    f'<span class="tag">{e(m.role)}</span><h3><a href="modules/{e(slug(m.path))}">{e(m.title)}</a></h3>'
                    f'<p>{e(m.method)}</p><div class="small muted">{e(m.path)}</div></article>' for m in prim)
    old_cards = ''.join(f'<article class="card" data-search="{e((m.path+" "+m.method).lower())}">'
                        f'<span class="tag badge-legacy">Legacy</span><h3><a href="modules/{e(slug(m.path))}">{e(m.title)}</a></h3>'
                        f'<p>{e(m.method)}</p></article>' for m in legacy)
    current = sum((root/m.path).is_file() for m in MODULES)
    content = f'''
    <div class="eyebrow">Engineering computation · Source reference · 2026</div>
    <h1>Predicting Building Occupancy<br>Using Machine Learning</h1>
    <p class="lead">Offline-readable HTML documentation for the MMA3001 regression workflow: extraction, numerical averaging, temporal features, model fitting, cross-room validation and report figure generation.</p>
    <div class="hint"><strong>Documentation method:</strong> This site records the script interfaces, data dependencies, outputs and known limitations from the GitHub project. The generator <strong>parses local Python source without importing it</strong>, so creating these pages does not run the multi-hour SVR training or modify datasets. Indexed local source modules: <strong>{current}/{len(MODULES)}</strong>.</div>
    <h2>Computational workflow</h2><div class="workflow">{steps}</div>
    <p>Models are fitted on <strong>room G.25</strong> and independently evaluated on <strong>room G.20</strong>. Both final targets use <strong>one-minute time-weighted headcount</strong> based on a zero-order-hold approximation of irregularly timed readings.</p>
    <div class="table-wrap"><table><thead><tr><th>Data feature</th><th>Role</th><th>Definition</th></tr></thead><tbody>
    <tr><td>Timestamp</td><td>Original input</td><td>Reading time; interpreted as UTC and converted to Australia/Melbourne</td></tr>
    <tr><td>Headcount</td><td>Target</td><td>One-minute time-weighted average of recorded occupancy (occupants; may be fractional)</td></tr>
    <tr><td>TimeSin, TimeCos</td><td>Model inputs</td><td>Sine/cosine representation of fractional hour of day</td></tr>
    <tr><td>DaySin, DayCos</td><td>Model inputs</td><td>Sine/cosine representation of day of week</td></tr>
    <tr><td>MonthSin, MonthCos</td><td>Model inputs</td><td>Sine/cosine representation of month of year</td></tr>
    <tr><td>IsWeekend</td><td>Model input</td><td>1 on Saturday/Sunday; 0 otherwise</td></tr></tbody></table></div>
    <h2>Module reference</h2>
    <input class="search" type="search" id="module-search" placeholder="Filter modules by name, stage or description..." aria-label="Filter source modules" oninput="document.querySelectorAll('[data-search]').forEach(x=>x.hidden=!x.dataset.search.includes(this.value.toLowerCase()))">
    <div class="cards">{cards}</div><h2>Earlier workflows (not final report pipeline)</h2><div class="cards">{old_cards}</div>
    <h2>Numerical verification</h2>
    <div class="card"><p><strong>19 automated pytest checks passed</strong> in the user's Windows environment on 26 October 2026, including 12 preprocessing cases, 4 feature-engineering checks and 3 exported G.20 metrics-consistency tests. This verifies selected implementation behaviours and consistency of saved metrics; it does not establish sensor ground-truth accuracy.</p>
    <p><a href="{e(REPO_URL)}/blob/main/docs/TEST_REPORT.md">Read the test report ↗</a> · <a href="{e(REPO_URL)}/tree/main/tests">Browse the test functions ↗</a></p></div>
    <h2>Reproduce or rebuild</h2>
    <p>From the project root, after installing dependencies and fetching data via Git LFS, execute:</p>
    <pre><code>python src/01_Extract_Occupancy_Data.py
python src/02_Inspect_Occupancy_Data.py
python src/03_preprocess_training.py
python src/03b_preprocess_G20.py
python src/04_feature_engineering.py
python src/04b_feature_engineering_G20.py
python src/05_train_models.py
python src/06b_validate_G20_minute.py
python src/07_generate_report_figures.py

# Regenerate these HTML pages from local source (fast; no imports)
python tools/build_html_docs.py</code></pre>
    <div class="warn"><strong>Before re-running cross-room evaluation:</strong> Script 05 currently saves the full SVR checkpoint as <code>05_svr_model_full.joblib</code>, while Script 06b looks up <code>05_svr_model.joblib</code>. Align the filenames to ensure the two reported evaluations correspond to the same fitted model.</div>
    <p class="muted small">Complete installation instructions, original dataset handling, and reproducibility caveats are in <a href="{e(REPO_URL)}/blob/main/README.md">README.md ↗</a>.</p>
    '''
    return layout('Overview and workflow','index',content)


def build(project_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True,exist_ok=True)
    modules_dir = output_dir/'modules'
    modules_dir.mkdir(parents=True,exist_ok=True)
    assets_dir = output_dir/'assets'
    assets_dir.mkdir(parents=True,exist_ok=True)
    (assets_dir/'style.css').write_text(CSS,encoding='utf-8')
    (output_dir/'index.html').write_text(index(project_root),encoding='utf-8')
    for m in MODULES:
        (modules_dir/slug(m.path)).write_text(render_module(m,project_root),encoding='utf-8')
    count = sum((project_root/m.path).is_file() for m in MODULES)
    print(f"Generated {len(MODULES)+1} HTML pages in {output_dir}")
    print(f"Production source modules found: {count}/{len(MODULES)}")
    if count<len(MODULES):
        print("NOTE: Re-run from the actual Git project root to embed full code listings and exact signatures.")
    print(f"Open: {output_dir/'index.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=None,help='Path to repository root. Default: parent of tools/.')
    parser.add_argument('--out', type=Path,default=None,help='Output directory. Default: <root>/docs/html/.')
    args=parser.parse_args()
    root = (args.root or Path(__file__).resolve().parent.parent).resolve()
    out = (args.out or (root/'docs'/'html')).resolve()
    build(root,out)


if __name__=='__main__':
    main()
