"""Read-only dashboard: individual data synthetic, external outputs aggregates only."""
from pathlib import Path
import json
import re
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from .word_export import export_docx, markdown_docx, MIME
from . import runs
from .oulad_external import REQUIRED_FILES, REQUIRED_COLUMNS
from pydantic import BaseModel
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "synthetic_demo.csv"
app = FastAPI(title="EduRisk Research — demonstration and aggregated external evaluation", version="0.8.0")

@app.middleware('http')
async def prevent_stale_ui(request: Request, call_next):
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    return response


MATERIAL_FILES = {
    'calibration-protocol': 'research/23_separate_calibration_protocol.md',
    'calibration-results': 'research/24_separate_calibration_results.md',
    'subgroup-results': 'research/21_subgroup_results.md',
    'six-models': 'research/22_six_models.md',
    'external-execution': 'research/19_external_execution_v09.md',
    'external-results': 'research/20_external_results_v09.md',

    'robustness-protocol': 'research/12_robustness_protocol_v05.md',
    'nested-protocol': 'research/14_nested_strategy_protocol_v06.md',
    'uncertainty-model': 'research/16_uncertainty_mathematical_model_v07.md',
    'external-protocol': 'research/18_external_protocol_v08.md',

    "problem": "research/01_problem.md",
    "literature": "research/02_literature.md",
    "data-protocol": "research/03_data_protocol.md",
    "roadmap": "research/04_roadmap.md",
    "history-method": "research/05_longitudinal_method.md",
    "history-results": "research/06_synthetic_results_v02.md",
    "window-protocol": "research/07_window_missingness_protocol_v03.md",
    "window-results": "research/08_window_missingness_results_v03.md",
    "adaptive-hypothesis": "research/09_scientific_hypothesis_next_stage.md",
    "adaptive-model": "research/10_adaptive_mathematical_model_v04.md",
    "adaptive-results": "research/11_adaptive_results_v04.md",
    "adaptive-table": "reports/adaptive_comparison_synthetic.csv",
    "window-table": "reports/ablation_aggregates_synthetic.csv",
    "adaptive-delta": "reports/adaptive_delta_ap_synthetic.png",
    "adaptive-windows": "reports/adaptive_selected_windows_synthetic.png",
}


def require_demo() -> dict:
    path = REPORTS / "summary.json"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Run generation/training first (see README.md)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("mode") != "SYNTHETIC_DEMO_ONLY":
        raise HTTPException(status_code=403, detail="Only synthetic demo reports can be served")
    return data


def _synthetic_frame(path: Path) -> pd.DataFrame:
    require_demo()
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Demo data not generated yet")
    df = pd.read_csv(path)
    if "student_id" not in df or not df["student_id"].astype(str).str.fullmatch(r"SYN-\d+").all():
        raise HTTPException(status_code=403, detail="Only synthetic IDs allowed in the local dashboard")
    return df


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mode": "SYNTHETIC_DEMO_ONLY", "version": "0.8.0"}


@app.get("/api/summary")
def summary() -> dict:
    return require_demo()


@app.get("/api/predictions")
def predictions() -> dict:
    frame = _synthetic_frame(REPORTS / "predictions_synthetic.csv")
    return {"mode": "SYNTHETIC_DEMO_ONLY", "total": len(frame), "rows": frame.to_dict(orient="records")}


@app.get("/api/term-trends")
def term_trends() -> dict:
    frame = _synthetic_frame(DATA)
    # This endpoint contains descriptive historical data, NOT predicted outcomes.
    groups = frame.groupby("term_index", sort=True)
    rows = [
        {"term_index": int(term), "n_students": int(len(chunk)),
         "mean_grade": round(float(chunk["current_avg_grade"].mean()), 4),
         "mean_attendance": round(float(chunk["attendance_rate"].mean()), 4),
         "share_current_below_3": round(float((chunk["current_avg_grade"] < 3.0).mean()), 4)}
        for term, chunk in groups
    ]
    return {"mode": "SYNTHETIC_DEMO_ONLY", "rows": rows}


@app.get("/api/students/{student_id}/trajectory")
def student_trajectory(student_id: str) -> dict:
    if re.fullmatch(r"SYN-\d+", student_id) is None:
        raise HTTPException(status_code=400, detail="Only synthetic student identifiers supported")
    frame = _synthetic_frame(DATA)
    subset = frame.loc[frame["student_id"] == student_id].sort_values("term_index")
    if subset.empty:
        raise HTTPException(status_code=404, detail="Synthetic ID not found")
    fields = ["term_index", "current_avg_grade", "attendance_rate",
              "assignment_completion_rate", "credits_completion_rate", "late_submission_rate"]
    report = _synthetic_frame(REPORTS / "predictions_synthetic.csv")
    selected = report.loc[report["student_id"] == student_id]
    forecast = None
    if not selected.empty:
        p = selected.iloc[0]
        forecast = {"source_term": int(p["term_index"]),
                    "next_term": int(p["term_index"]) + 1,
                    "risk_probability": float(p["risk_probability"])}
    return {"mode": "SYNTHETIC_DEMO_ONLY", "student_id": student_id,
            "observations": subset[fields].to_dict(orient="records"), "forecast": forecast}


@app.get("/api/history-ablation")
def history_ablation() -> dict:
    """Read-only results of the separate 10-term synthetic experiment."""
    summary_path = REPORTS / "ablation_summary_synthetic.json"
    aggregate_path = REPORTS / "ablation_aggregates_synthetic.csv"
    if not summary_path.is_file() or not aggregate_path.is_file():
        raise HTTPException(status_code=503, detail="Run python -m edurisk.ablation first")
    experiment = json.loads(summary_path.read_text(encoding="utf-8"))
    if experiment.get("mode") != "SYNTHETIC_DEMO_ONLY":
        raise HTTPException(status_code=403, detail="Only synthetic experiment results allowed")
    aggregate = pd.read_csv(aggregate_path)
    # None instead of NaN: JSON is strictly serializable and safe for browser.
    rows = json.loads(aggregate.to_json(orient="records"))
    return {"mode": "SYNTHETIC_DEMO_ONLY", "version": "0.3.0",
            "design": experiment["design"], "limitations": experiment["limitations"],
            "window_selections": experiment["window_selections"], "rows": rows}


@app.get("/api/adaptive-history")
def adaptive_history() -> dict:
    """Aggregated synthetic benchmark for v0.4, not per-student decisions."""
    path = REPORTS / "adaptive_summary_synthetic.json"
    usage_path = REPORTS / "adaptive_window_usage_synthetic.csv"
    if not path.is_file() or not usage_path.is_file():
        raise HTTPException(status_code=503, detail="Run python -m edurisk.adaptive first")
    experiment = json.loads(path.read_text(encoding="utf-8"))
    if experiment.get("mode") != "SYNTHETIC_DEMO_ONLY":
        raise HTTPException(status_code=403, detail="Only synthetic experiment results allowed")
    usage = pd.read_csv(usage_path)
    return {"mode": "SYNTHETIC_DEMO_ONLY", "version": "0.4.0",
            "design": experiment["design"], "formula": experiment["formula"],
            "limitations": experiment["limitations"],
            "selections": experiment["selections"],
            "comparison": experiment["comparison"],
            "window_usage": json.loads(usage.to_json(orient="records"))}


@app.get("/api/robustness")
def robustness() -> dict:
    """Read-only v0.5 summary for synthetic forward/cohort robustness experiments."""
    path = REPORTS / "robustness_summary_synthetic.json"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Run python -m edurisk.robustness first")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("mode") != "SYNTHETIC_DEMO_ONLY":
        raise HTTPException(status_code=403, detail="Only synthetic experiment results allowed")
    return {"mode":report["mode"],"version":report["version"],
            "design":report["design"],"protocols":report["protocols"],
            "n_trained_models":report["n_trained_models"],
            "n_folds":report["n_folds"],"limitations":report["limitations"],
            "aggregate_results":report["aggregate_results"],
            "comparative_deltas":report["comparative_deltas"],
            "mask_realizations":report["mask_realizations"]}



@app.get("/api/nested-selection")
def nested_selection() -> dict:
    """Aggregates only: do not expose any real learner profiles or individual decisions."""
    source = REPORTS / "nested_summary_synthetic.json"
    if not source.is_file():
        raise HTTPException(status_code=503, detail="Run python -m edurisk.nested_selection first")
    result=json.loads(source.read_text(encoding="utf-8"))
    if result.get("mode")!="SYNTHETIC_DEMO_ONLY":
        raise HTTPException(status_code=403, detail="Only synthetic reports permitted")
    return result

@app.get("/api/uncertainty-selection")
def uncertainty_selection() -> dict:
    """Only aggregated synthetic research outputs, no individual student forecasts."""
    source = REPORTS / "uncertainty_summary_synthetic.json"
    if not source.exists():
        raise HTTPException(status_code=503, detail="Run python -m edurisk.uncertainty_selection first")
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get("mode") != "SYNTHETIC_DEMO_ONLY":
        raise HTTPException(status_code=403, detail="Only synthetic research data permitted")
    return result


@app.get("/api/external-validation")
def external_validation(dataset: str = Query('fixture', pattern='^(fixture|oulad)$'),
                        protocol: str = Query('shared', pattern='^(shared|separate)$'),
                        family: str = Query('logistic_regression', pattern='^(logistic_regression|random_forest|hist_gradient_boosting|extra_trees|knn|gaussian_nb)$')) -> dict:
    """Explicit dataset selection: real results never fall back to the fixture."""
    run_id = None
    if dataset == 'oulad' or protocol == 'separate' or family not in ('logistic_regression','random_forest'):
        profile = next(k for k,v in runs.PROFILES.items() if v == family and k.startswith('oulad_' if dataset == 'oulad' else 'fixture_') and k.endswith('_separate') == (protocol=='separate'))
        record = next((r for r in runs.history(limit=None) if r['profile'] == profile and r['status'] == 'completed'), None)
        path = runs.RUNS / record['id'] / 'external_oulad_summary.json' if record else None
        run_id = record['id'] if record else None
    else:
        path = REPORTS / ('external_oulad_summary.json' if family == 'logistic_regression' else 'fixture_random_forest/external_oulad_summary.json')
    if path is None or not path.is_file():
        return {'mode':'NOT_RUN', 'version':'0.8.0', 'message':'No completed report for this dataset and family'}
    result = json.loads(path.read_text(encoding='utf-8'))
    expected = 'REAL_OULAD_EXTERNAL' if dataset == 'oulad' else 'SYNTHETIC_SCHEMA_FIXTURE'
    if result.get('mode') != expected or result.get('design', {}).get('family') != family or result.get('design',{}).get('calibration_protocol','shared') != protocol:
        raise HTTPException(403, 'Report provenance or family mismatch')
    allowed = ('mode','version','source','design','n_rows','classes','selection','metrics',
                 'comparison','bootstrap','selective','limitations','provenance','environment','cohort_audit','subgroup_audit','calibration_audit')
    return {**{key:result[key] for key in allowed if key in result}, 'run_id':run_id}


@app.get("/api/research-plan")
def research_plan() -> dict:
    """Source-derived schedule; completion belongs to the user's local tracker."""
    path = ROOT / "research" / "plan.json"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Research plan is unavailable")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/materials/{material_id}")
def research_material(material_id: str, format: str = Query('docx', pattern='^(source|docx)$')):
    """Serve only named public research notes and synthetic aggregate results."""
    relative = MATERIAL_FILES.get(material_id)
    if relative is None:
        raise HTTPException(status_code=404, detail="Unknown research material")
    path = (ROOT / relative).resolve()
    # Resolve junctions/symlinks as well as dot segments before reading anything.
    project_root = ROOT.resolve()
    allowed_roots = [project_root / directory for directory in ("research", "reports")]
    if not any(path.is_relative_to(directory) for directory in allowed_roots):
        raise HTTPException(status_code=404, detail="Research material is unavailable")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Research material is unavailable")
    if format == 'docx':
        return Response(export_docx(path), media_type=MIME, headers={'Content-Disposition':f'attachment; filename="{path.stem}.docx"'})
    media_type = {".md": "text/plain; charset=utf-8", ".csv": "text/csv; charset=utf-8",
                  ".png": "image/png"}[path.suffix]
    return FileResponse(path, media_type=media_type, filename=path.name,
                        content_disposition_type="inline",
                        headers={"X-Content-Type-Options": "nosniff"})





class RunRequest(BaseModel):
    profile: str


@app.get('/api/development')
def development():
    directory = ROOT / 'data' / 'oulad' / 'raw'
    files = []
    for name in REQUIRED_FILES:
        path = directory / name
        missing = []
        error = None
        if path.is_file():
            try:
                missing = sorted(REQUIRED_COLUMNS[name] - set(pd.read_csv(path, nrows=0).columns))
            except Exception:
                error = 'Не удалось прочитать заголовок CSV'
        files.append({'name':name, 'exists':path.is_file(), 'missing_columns':missing, 'error':error})
    return {'version':'0.8.0', 'files':files,
            'schema_ready':all(f['exists'] and not f['missing_columns'] and not f['error'] for f in files),
            'profiles':list(runs.PROFILES), 'runs':runs.history()}


@app.post('/api/runs', status_code=202)
def start_run(payload: RunRequest, request: Request):
    # Same-origin JSON requests only; no cross-site form can start computation.
    if request.headers.get('sec-fetch-site') == 'cross-site':
        raise HTTPException(403, 'Cross-site run requests are not allowed')
    origin = request.headers.get('origin')
    if origin and (urlsplit(origin).netloc != request.headers.get('host') or urlsplit(origin).scheme != request.url.scheme):
        raise HTTPException(403, 'Origin does not match local application')
    try:
        return runs.start(payload.profile)
    except ValueError as error:
        raise HTTPException(400, str(error))
    except RuntimeError as error:
        raise HTTPException(409, str(error))


@app.get('/api/runs')
def run_history():
    return {'runs':runs.history()}


@app.get('/api/runs/{run_id}/{artifact}')
def run_artifact(run_id: str, artifact: str, format: str = Query('source', pattern='^(source|docx)$')):
    if not re.fullmatch(r'[0-9a-f]{32}', run_id):
        raise HTTPException(404, 'Unknown run')
    names = {'summary':'external_oulad_summary.json', 'table':'external_oulad_aggregates.csv', 'log':'run.log'}
    if artifact not in names:
        raise HTTPException(404, 'Unknown artifact')
    path = runs.RUNS / run_id / names[artifact]
    if not path.is_file():
        raise HTTPException(404, 'Artifact not ready')
    if format == 'docx':
        return Response(export_docx(path), media_type=MIME, headers={'Content-Disposition':f'attachment; filename="{artifact}-{run_id[:8]}.docx"'})
    media = {'summary':'application/json', 'table':'text/csv', 'log':'text/plain'}[artifact]
    return FileResponse(path, media_type=media, filename=path.name, headers={'X-Content-Type-Options':'nosniff'})


class PlanExport(BaseModel):
    months: dict[str, dict[str, str]] = {}

@app.post('/api/word/plan')
def word_plan(payload: PlanExport):
    if len(payload.months)>36 or any(len(str(v))>20000 for v in payload.months.values()):
        raise HTTPException(413,'Plan notes too large')
    plan=json.loads((ROOT/'research/plan.json').read_text(encoding='utf-8'))
    text='# План диссертации и отметки выполнения\n\n'+plan['title']+'\n\nОтметки переданы из браузера пользователем.\n\n'
    for month in plan['months']:
        entry=payload.months.get(month['id'],{})
        status={'planned':'Запланировано','active':'В работе','done':'Выполнено'}.get(entry.get('status'),'Запланировано')
        note=re.sub(r'([\\`*_{}\[\]()<>#!|$])',r'\\\1',entry.get('note',''))
        text+='## '+month['label']+'\n\n'+status+'\n\n'+'\n\n'.join(month['tasks']+month['deliverables'])+'\n\n'+note+'\n\n'
    return Response(markdown_docx(text),media_type=MIME,headers={'Content-Disposition':'attachment; filename="research-plan.docx"'})

app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")
