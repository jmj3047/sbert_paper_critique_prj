"""Partial Table 1 replication and biomedical zero-shot evaluation. No training."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'stsb': ('sentence-transformers/stsb', 'test', 1379, 5.0),
    'biosses': ('tabilab/biosses', 'train', 100, 1.0),
}
MODELS = {'bert': 'google-bert/bert-base-uncased',
          'sbert': 'sentence-transformers/bert-base-nli-mean-tokens'}
EBM_CATEGORIES = ('background', 'intervention', 'outcome', 'population', 'study_design')


def parse_ebmsass(payload, category):
    rows = []
    for number, line in enumerate(payload.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split('\t')
        if len(fields) != 4:
            raise ValueError(f'EBMSASS {category}:{number}: expected four tab-separated fields')
        pair_id, sentence1, sentence2, score = fields
        if not pair_id or not sentence1.strip() or not sentence2.strip() or not 1 <= float(score) <= 5:
            raise ValueError(f'Invalid EBMSASS row {category}:{number}')
        rows.append(dict(pair_id=pair_id, category=category, sentence1=sentence1,
                         sentence2=sentence2, score=float(score)))
    return rows


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def prepare_data(selected):
    directory = ROOT / 'data'
    directory.mkdir(exist_ok=True)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in selected:
        if name in manifest:
            for file in manifest[name]['files']:
                actual = hashlib.sha256((directory / file['local']).read_bytes()).hexdigest()
                if actual != file['sha256']:
                    raise ValueError('Data checksum mismatch: ' + file['local'])
            continue
        if name == 'ebmsass':
            repo = 'HamedHassanzadeh/EBMSASS'
            revision = json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/master'))['sha']
            files = []
            for category in EBM_CATEGORIES:
                url = f'https://raw.githubusercontent.com/{repo}/{revision}/{category}_pairs.txt'
                payload = fetch(url)
                rows = parse_ebmsass(payload, category)
                if len(rows) != 200:
                    raise ValueError(f'Expected 200 {category} pairs, got {len(rows)}')
                local = f'ebmsass-{category}.txt'
                (directory / local).write_bytes(payload)
                files.append({'local': local, 'url': url, 'category': category,
                              'sha256': hashlib.sha256(payload).hexdigest()})
            manifest[name] = {'repo': repo, 'revision': revision, 'split': 'all_zero_shot',
                              'format': 'ebmsass_tsv', 'expected_rows': 1000,
                              'gold_scale': 1.0, 'files': files}
            write_json(manifest_path, manifest)
            print(f'Downloaded {name}: revision {revision}', flush=True)
            continue
        repo, split, count, scale = SOURCES[name]
        info = json.loads(fetch('https://huggingface.co/api/datasets/' + repo))
        revision = info['sha']
        paths = [x['rfilename'] for x in info['siblings']
                 if x['rfilename'].startswith('data/' + split + '-')
                 and x['rfilename'].endswith('.parquet')]
        if not paths:
            raise ValueError('No expected parquet files found for ' + repo)
        files = []
        for i, remote in enumerate(sorted(paths)):
            url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}'
            payload = fetch(url)
            local = f'{name}-{i}.parquet'
            (directory / local).write_bytes(payload)
            files.append({'local': local, 'url': url,
                          'sha256': hashlib.sha256(payload).hexdigest()})
        card = fetch(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/README.md')
        (directory / f'{name}_dataset_card.md').write_bytes(card)
        manifest[name] = {'repo': repo, 'revision': revision, 'split': split,
                          'expected_rows': count, 'gold_scale': scale, 'files': files}
        write_json(manifest_path, manifest)
        print(f'Downloaded {name}: revision {revision}', flush=True)
    return {name: manifest[name] for name in selected}


def correlations(gold, prediction):
    import numpy as np
    from scipy.stats import spearmanr, pearsonr
    if np.ptp(gold) == 0 or np.ptp(prediction) == 0:
        return [float('nan'), float('nan')]
    return [float(spearmanr(gold, prediction).statistic),
            float(pearsonr(gold, prediction).statistic)]


def bootstrap(gold, predictions, iterations, seed):
    """Shared resampled row indices preserve pairing between model predictions."""
    import numpy as np
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in predictions}
    for _ in range(iterations):
        ix = rng.integers(0, len(gold), size=len(gold))
        for name, pred in predictions.items():
            samples[name].append(correlations(gold[ix], pred[ix]))
    arrays = {k: np.asarray(v) for k, v in samples.items()}

    def interval(array):
        valid = array[np.isfinite(array).all(axis=1)]
        if not len(valid):
            return {'valid_resamples': 0, 'spearman': None, 'pearson': None}
        bounds = np.percentile(valid, [2.5, 97.5], axis=0)
        return {'valid_resamples': len(valid), 'spearman': bounds[:, 0].tolist(),
                'pearson': bounds[:, 1].tolist()}
    cis = {k: interval(v) for k, v in arrays.items()}
    differences = {}
    for baseline in ('bert_mean', 'bert_cls'):
        differences['sbert_minus_' + baseline] = {
            'estimate': (np.asarray(correlations(gold, predictions['sbert_mean'])) -
                         np.asarray(correlations(gold, predictions[baseline]))).tolist(),
            'ci95': interval(arrays['sbert_mean'] - arrays[baseline])}
    return cis, differences


def encode(texts, model, tokenizer, device, batch_size, max_length):
    import numpy as np
    import torch
    means, clses = [], []
    lengths = [len(x) for x in tokenizer(texts, truncation=False,
                                        verbose=False)['input_ids']]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True,
                              truncation=True, max_length=max_length, return_tensors='pt')
            batch = {k: v.to(device) for k, v in batch.items()}
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            means.append(mean.cpu().numpy())
            clses.append(hidden[:, 0].cpu().numpy())
    return np.concatenate(means), np.concatenate(clses), {
        'unique_sentences': len(texts), 'truncated_sentences': sum(n > max_length for n in lengths),
        'truncation_rate': sum(n > max_length for n in lengths) / len(texts),
        'max_untruncated_tokens': max(lengths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=[*SOURCES, 'ebmsass'],
                        default=list(SOURCES), help='Default: stsb biosses; add ebmsass explicitly')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, help='Smoke test only; NOT a reportable full evaluation')
    args = parser.parse_args()
    if min(args.batch_size, args.max_length, args.bootstrap) < 1 or (args.limit is not None and args.limit < 3):
        parser.error('Positive settings required; --limit must be at least 3')
    manifest = prepare_data(list(dict.fromkeys(args.datasets)))
    if args.download_only:
        return
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModel, AutoTokenizer
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = ROOT / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True)
    lock_path = ROOT / 'model_revisions.json'
    revisions = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    for name, repo in MODELS.items():
        if name not in revisions:
            revisions[name] = {'repo': repo, 'revision': HfApi().model_info(repo).sha}
            write_json(lock_path, revisions)
        if revisions[name]['repo'] != repo:
            raise ValueError('Model lock does not match configured model')
    metadata = {'arguments': vars(args), 'platform': platform.platform(),
                'python': platform.python_version(), 'models': revisions, 'datasets': manifest,
                'smoke_test_only': args.limit is not None,
                'versions': {p: importlib.metadata.version(p) for p in
                             ['torch', 'transformers', 'huggingface-hub', 'numpy', 'scipy', 'pyarrow']}}
    write_json(output / 'run_manifest.json', metadata)
    datasets, predictions, truncation = {}, {}, {}
    for name, source in manifest.items():
        rows = []
        for file in source['files']:
            path = ROOT / 'data' / file['local']
            if source.get('format') == 'ebmsass_tsv':
                rows.extend(parse_ebmsass(path.read_bytes(), file['category']))
            else:
                rows.extend(pq.read_table(path).to_pylist())
        assert len(rows) == source['expected_rows'], (name, len(rows))
        if args.limit:
            rows = rows[:args.limit]
        for row in rows:
            row['gold'] = float(row['score']) * source['gold_scale']
            lower, upper = (1, 5) if name == 'ebmsass' else (0, 4 if name == 'biosses' else 5)
            if not lower <= row['gold'] <= upper:
                raise ValueError(f'Invalid {name} gold score: {row["gold"]}')
        datasets[name] = rows
        predictions[name], truncation[name] = {}, {}
    for kind, info in revisions.items():
        print('Loading ' + info['repo'], flush=True)
        tokenizer = AutoTokenizer.from_pretrained(info['repo'], revision=info['revision'])
        model = AutoModel.from_pretrained(info['repo'], revision=info['revision'],
                                          use_safetensors=True).to(args.device)
        for name, rows in datasets.items():
            texts = list(dict.fromkeys(s for row in rows for s in (row['sentence1'], row['sentence2'])))
            index = {s: i for i, s in enumerate(texts)}
            mean, cls, stats = encode(texts, model, tokenizer, args.device,
                                      args.batch_size, args.max_length)
            truncation[name][kind] = stats
            for pooling, embeddings in [('mean', mean)] + ([('cls', cls)] if kind == 'bert' else []):
                embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
                predictions[name][kind + '_' + pooling] = np.array([
                    np.dot(embeddings[index[r['sentence1']]], embeddings[index[r['sentence2']]])
                    for r in rows])
            print(f'Encoded {kind}/{name}: {len(rows)} pairs', flush=True)
        del model
        if args.device == 'mps':
            torch.mps.empty_cache()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    summary = []
    for name, rows in datasets.items():
        pred = predictions[name]
        gold = np.array([r['gold'] for r in rows])
        with (output / f'{name}_predictions.csv').open('w', newline='') as stream:
            extra_fields = ['pair_id', 'category'] if name == 'ebmsass' else []
            writer = csv.DictWriter(stream, fieldnames=['row_id', 'sentence1', 'sentence2', 'gold'] + extra_fields + list(pred))
            writer.writeheader()
            for i, r in enumerate(rows):
                writer.writerow({'row_id': i, 'sentence1': r['sentence1'], 'sentence2': r['sentence2'],
                                 'gold': r['gold'], **{k: r[k] for k in extra_fields},
                                 **{k: float(v[i]) for k, v in pred.items()}})
        cis, differences = bootstrap(gold, pred, args.bootstrap, args.seed)
        write_json(output / f'{name}_paired_differences.json', differences)
        for model_name, scores in pred.items():
            rho, r = correlations(gold, scores)
            paper = {'bert_mean': 46.35, 'bert_cls': 16.50, 'sbert_mean': 77.03}.get(model_name) if name == 'stsb' and not args.limit else None
            summary.append({'dataset': name, 'model': model_name, 'n': len(rows),
                            'spearman': rho, 'pearson': r, 'ci95': cis[model_name],
                            'paper_spearman_x100': paper,
                            'difference_from_paper_points': rho * 100 - paper if paper is not None else None})
    write_json(output / 'summary.json', summary)
    write_json(output / 'truncation.json', truncation)
    lines = ['# Results' + (' — SMOKE TEST ONLY' if args.limit else ''), '',
             '| Dataset | Model | n | Spearman ×100 | Pearson ×100 |', '|---|---|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['n']} | {row['spearman']*100:.2f} | {row['pearson']*100:.2f} |")
    (output / 'results.md').write_text('\n'.join(lines) + '\n')
    print('Results: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
