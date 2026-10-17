#!/usr/bin/env python3
"""
scripts/generate_evidence.py
Genera EVIDENCE.md y actualiza CHANGELOG.md con un resumen de los mejores runs por experimento
Requiere: mlflow (el mismo que usas para tracking)
Uso:
  python scripts/generate_evidence.py --mlflow-uri file:./mlruns
Si no se pasa --mlflow-uri usa la variable de entorno MLFLOW_TRACKING_URI o file:./mlruns por defecto.
"""
import os
import argparse
from mlflow.tracking import MlflowClient
import json
from datetime import datetime


def select_best_run(runs):
    # Selecciona run con mejor 'f1' si existe, sino por 'accuracy', sino por primera métrica disponible
    best = None
    best_score = None
    for r in runs:
        metrics = r.data.metrics
        score = None
        if 'f1' in metrics:
            score = metrics['f1']
        elif 'accuracy' in metrics:
            score = metrics['accuracy']
        elif len(metrics) > 0:
            # pick first metric
            score = list(metrics.values())[0]
        if score is None:
            continue
        if best is None or score > best_score:
            best = r
            best_score = score
    return best


def format_run_summary(run):
    d = {
        'run_id': run.info.run_id,
        'status': run.info.status,
        'start_time': datetime.fromtimestamp(run.info.start_time/1000).isoformat() if run.info.start_time else None,
        'end_time': datetime.fromtimestamp(run.info.end_time/1000).isoformat() if run.info.end_time else None,
        'params': dict(run.data.params),
        'metrics': dict(run.data.metrics),
        'tags': dict(run.data.tags)
    }
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow-uri', type=str, default=None, help='MLflow tracking URI (e.g. file:./mlruns)')
    parser.add_argument('--out', type=str, default='EVIDENCE.md', help='Output markdown file')
    args = parser.parse_args()

    mlflow_uri = args.mlflow_uri or os.environ.get('MLFLOW_TRACKING_URI') or 'file:./mlruns'
    client = MlflowClient(tracking_uri=mlflow_uri)

    experiments = client.list_experiments()
    if not experiments:
        print('No experiments found at', mlflow_uri)

    lines = []
    lines.append('# EVIDENCE - resumen de mejores runs')
    lines.append('')
    lines.append(f'Tracking URI: {mlflow_uri}')
    lines.append('')

    for exp in experiments:
        lines.append(f'## Experiment: {exp.name} (id={exp.experiment_id})')
        # fetch runs
        runs = client.search_runs([exp.experiment_id], order_by=['attributes.start_time DESC'], max_results=1000)
        if not runs:
            lines.append('No runs found for this experiment.')
            lines.append('')
            continue
        best = select_best_run(runs)
        if best is None:
            lines.append('No metric (f1/accuracy/...) found in runs to select best run.')
            lines.append('')
            continue
        summary = format_run_summary(best)
        lines.append(f"- Best run_id: `{summary['run_id']}`")
        lines.append(f"  - status: {summary['status']}")
        lines.append(f"  - start_time: {summary['start_time']}")
        lines.append(f"  - end_time: {summary['end_time']}")
        lines.append('  - params:')
        for k,v in summary['params'].items():
            lines.append(f"    - {k}: {v}")
        lines.append('  - metrics:')
        for k,v in summary['metrics'].items():
            lines.append(f"    - {k}: {v}")
        # list artifacts path (local file paths under mlruns)
        run_artifact_uri = client.get_run(summary['run_id']).info.artifact_uri
        lines.append(f"  - artifact_uri: {run_artifact_uri}")
        lines.append('')

    # write EVIDENCE.md
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('Wrote', args.out)

    # append short summary to CHANGELOG.md
    changelog_path = 'CHANGELOG.md'
    changelog_lines = ['','## Evidence generated','', f'- Generated EVIDENCE at {datetime.utcnow().isoformat()} UTC', '']
    try:
        with open(changelog_path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(changelog_lines))
        print('Appended note to', changelog_path)
    except Exception as e:
        print('Could not update CHANGELOG.md:', e)

if __name__ == '__main__':
    main()
