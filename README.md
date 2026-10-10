# Sentence-BERT: Partial Reproduction and Biomedical Evaluation

This repository contains evaluation code and final results for a critique of Reimers and Gurevych's **Sentence-BERT (SBERT)** paper. It partially reproduces the STS Benchmark (STS-B) evaluation in Table 1, then evaluates the same frozen models on biomedical sentence similarity data.

BIOSSES provides an initial, small-sample evaluation. **EBMSASS is the main domain-extension experiment**, with 1,000 sentence pairs. No model is fine-tuned on these biomedical datasets.

## Models and research question

**Does the advantage of an NLI-trained SBERT over ordinary BERT sentence embeddings persist on biomedical similarity data without target-domain fine-tuning?**

| Method | Checkpoint | Sentence representation |
|---|---|---|
| BERT mean | `google-bert/bert-base-uncased` | Mean of final-layer token representations, excluding padding |
| BERT CLS | Same BERT checkpoint | Raw final-layer CLS representation, not `pooler_output` |
| SBERT mean | `sentence-transformers/bert-base-nli-mean-tokens` | Mean pooling after Natural Language Inference (NLI) training |

The BERT baselines follow the paper's mean/CLS evaluation approach. Exact identity with the original baseline checkpoint has not been established. This is an **evaluation-only partial reproduction**, not a reproduction of NLI training or all experiments in the paper.

The SBERT checkpoint is deliberately retained for the historical comparison. Its current model card marks it as deprecated; this repository does not recommend it as the best current production embedding model.

## Installation

Use Python 3.10. An isolated environment is recommended for a new installation. Run commands from the repository directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If using an existing compatible environment, activate it instead; creating another environment is optional.

**Environment caveat:** `requirements.txt` contains dependency pins that differ from the recorded final runs. Installing it does not recreate the exact environment used for the published results. The final `run_manifest.json` files record Python 3.10.13, PyTorch 2.13.0, Transformers 4.44.2, huggingface-hub 0.36.2, NumPy 1.26.4, SciPy 1.13.1, and PyArrow 21.0.0. Consult these manifests when investigating numerical differences; the old `environment.lock.txt` is not the authoritative record.

The final runs used Apple Metal Performance Shaders (MPS). Check availability with:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Use `--device cpu` if MPS is unavailable, or `--device cuda` on a compatible NVIDIA setup. CPU is the script's default.

## Download data locally

**Raw datasets and model weights are not included in this repository.** Internet access is needed for the initial downloads.

```bash
python run_experiment.py --datasets stsb biosses ebmsass --download-only
```

The script downloads data to `data/` and records source revisions and SHA-256 checksums. Model files are downloaded through Hugging Face during inference; model revisions are fixed in `model_revisions.json`.

On a fresh data download, the script resolves the current upstream data revision, rather than automatically using the revision in a historical results manifest. Compare the generated `data/manifest.json` with the final run manifests before claiming exact data-version reproduction. Subsequent runs reuse the local data cache and verify checksums.

## Run the experiments

First perform a small smoke test:

```bash
python run_experiment.py --datasets stsb biosses ebmsass --device mps --limit 8 --bootstrap 50
```

The limited run checks execution only; **do not report it as a full evaluation**.

To match the two final evaluation groups:

```bash
python run_experiment.py --datasets stsb biosses --device mps --bootstrap 2000
python run_experiment.py --datasets ebmsass --device mps --bootstrap 2000
```

Alternatively, evaluate all three datasets together:

```bash
python run_experiment.py --datasets stsb biosses ebmsass --device mps --bootstrap 2000
```

Each inference run creates a new timestamped results directory without overwriting previous results.

## Evaluation protocol

- **STS-B:** 1,379 test pairs. Downloaded labels on a 0–1 scale are multiplied by five to recover the 0–5 scale.
- **BIOSSES:** all 100 pairs, with 0–4 labels. Although the distributed split is named `train`, it is used only for evaluation here.
- **EBMSASS:** all 1,000 pairs, with 1–5 labels: 200 each for background, intervention, outcome, population, and study design.
- Frozen encoders in evaluation/inference mode; no additional training or label-based tuning.
- Maximum length 128 tokens and batch size 16 for all methods. Padding is excluded from mean pooling; special tokens are included. No manual lowercasing or stop-word removal is applied.
- Cosine similarity produces the predictions. Spearman correlation is the primary metric; Pearson correlation is secondary.
- Model differences use paired percentile bootstrap confidence intervals: 2,000 resamples, seed 42, and identical sampled pair indices for both models.
- Truncation is recorded for unique sentences. Download/inference timings are not treated as a controlled speed benchmark.

## Final results

All correlations below are multiplied by 100 for readability, matching the original paper's convention. **These values are not accuracy percentages.**

| Dataset | Pairs | BERT mean: Spearman | BERT CLS: Spearman | SBERT: Spearman | SBERT: Pearson |
|---|---:|---:|---:|---:|---:|
| STS-B | 1,379 | 47.29 | 20.31 | 76.99 | 74.14 |
| BIOSSES | 100 | 54.50 | 63.43 | 51.99 | 51.70 |
| EBMSASS | 1,000 | 51.89 | 51.14 | 56.32 | 60.18 |

The original paper reports STS-B Spearman scores of 46.35 for BERT mean, 16.50 for BERT CLS, and 77.03 for SBERT-NLI-base. The reproduced SBERT score is close to the reported value, and the large improvement over the baselines is retained.

| Dataset | SBERT − BERT mean | 95% confidence interval | SBERT − BERT CLS | 95% confidence interval |
|---|---:|---|---:|---|
| STS-B | +29.70 | [25.46, 34.04] | +56.68 | [51.51, 62.15] |
| BIOSSES | −2.51 | [−17.65, 13.19] | −11.44 | [−29.32, 5.18] |
| EBMSASS | +4.43 | [0.10, 8.78] | +5.18 | [0.02, 10.12] |

Differences and intervals were calculated before rounding. BIOSSES does not establish an advantage or inferiority. EBMSASS shows a modest positive advantage under this procedure, but its interval lower bounds are close to zero.

### Retained result directories

- `results/20261009T144708350843Z/`: full STS-B and BIOSSES evaluation.
- `results/20261009T145811500598Z/`: full EBMSASS evaluation.

Each retained directory contains:

| File | Purpose |
|---|---|
| `results.md` | Readable results table |
| `summary.json` | Numerical evaluation summary |
| `run_manifest.json` | Arguments, platform, package versions, model/data revisions and data checksums |
| `*_paired_differences.json` | Paired model differences and confidence intervals |
| `truncation.json` | Input truncation statistics |

The script also generates `*_predictions.csv` locally. These files contain original sentences and gold labels, so they are **excluded from the intended upload**, along with `data/`, model caches, and smoke-test results.

## Tests and reproducibility limits

Run the offline data-path tests without downloading models:

```bash
python -m unittest test_experiment_data.py
```

These tests check EBMSASS parsing, invalid records, cache reuse, and checksum verification. They do not validate encoder inference or reproduce the paper's scores. An automated continuous-integration workflow is not included.

- Historical runs recorded model/data revisions, settings, package versions, and pair-level predictions locally, but did not record an execution-code Git commit. A later repository commit does not retroactively establish the exact code state of those runs.
- Ordinary BERT and SBERT differ in NLI supervision as well as their use in the experiment; the comparison does not isolate the causal effect of Siamese architecture.
- Dataset content, label distributions, and annotation procedures differ. Cross-dataset score changes cannot be attributed solely to domain shift or sample size.
- Pair-level bootstrap intervals do not account for shared sentences/documents, training-seed variation, or multiple-comparison correction.
- Spearman and Pearson do not measure calibrated score errors or retrieval relevance. Cross-encoder speed and production search performance were not reproduced.
- Evaluation of all BIOSSES/EBMSASS pairs differs from held-out splits in other studies. Published results from those protocols are not directly comparable.
- Future domain fine-tuning should separate training, validation, and testing. These results have already been inspected, so independent unseen evaluation data would strengthen subsequent claims.

## Sources and related work

- Reimers and Gurevych (2019), [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/). DOI: 10.18653/v1/D19-1410.
- [BERT checkpoint, model card and usage examples](https://huggingface.co/google-bert/bert-base-uncased).
- [SBERT checkpoint, model card and mean-pooling example](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens).
- [Original Sentence Transformers code repository](https://github.com/UKPLab/sentence-transformers). This evaluation script uses Transformers `AutoModel` directly, not the Sentence Transformers wrapper.
- [STS-B distribution](https://huggingface.co/datasets/sentence-transformers/stsb).
- Soğancıoğlu, Öztürk, and Özgür (2017), [BIOSSES](https://doi.org/10.1093/bioinformatics/btx238); [dataset distribution](https://huggingface.co/datasets/tabilab/biosses).
- Hassanzadeh, Nguyen, and Verspoor (2019), [Quantifying semantic similarity of clinical evidence in the biomedical literature to facilitate related evidence synthesis](https://doi.org/10.1016/j.jbi.2019.103321); [author-released EBMSASS data](https://github.com/HamedHassanzadeh/EBMSASS).
- Wang et al. (2022), [Uncertainty Estimation and Reduction of Pre-trained Models for Text Regression](https://aclanthology.org/2022.tacl-1.39/). Its general-domain transfer and target-domain fine-tuning conditions motivate follow-up experiments; they were not reproduced here.
- Kanakarajan et al. (2022), [BioSimCSE: BioMedical Sentence Embeddings using Contrastive Learning](https://aclanthology.org/2022.louhi-1.10/). A domain-specific representation-learning alternative, not a model evaluated in these runs.

Check the upstream licenses and attribution requirements before redistributing datasets or model files. This repository's intended upload does not include the datasets.
