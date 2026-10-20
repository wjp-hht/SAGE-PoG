# SAGE-PoG: Efficient Multi-hop Graph Reasoning via Semantic Pre-filtering with Sufficiency-Aware Fallback

Reference implementation and derived results for the SAGE-PoG manuscript.
The manuscript is currently under review; bibliographic details will be added
here once it is published.

## What this repository contains

| Path | Contents |
|---|---|
| `src/` | Core implementation: PoG pipeline, the two pluggable modules (`pog_improvements.py`: relation Top-K pre-filtering and knowledge-sufficiency early stopping), LLM calling layer with HTTP/token/retry instrumentation, Freebase helpers |
| `run_experiment.py` | Driver for the four-group ablation (A/B/C/D) reported in the paper |
| `rebuild_military_kg.py` | Domain subgraph construction (N-Triples output) |
| `experiments/` | Launch scripts for the three main batches (G1/G2/G3) and the K / d_min sensitivity sweeps |
| `analysis/` | Result aggregation and figure generation |
| `configs/` | Configuration (hyper-parameters and paths) |
| `results/` | **Derived** result files used in the paper (not raw per-question logs); see the layout table below |
| `figures/` | Figure 1-4 as they appear in the paper (English-labelled) |
| `make_figures_en.py`, `figdata/` | Script and pre-extracted values used to draw Figures 1-4 |

## Environment

- Python 3.10+
- A locally deployed **Qwen3-32B** served through **Ollama** with the
  OpenAI-compatible `chat/completions` endpoint (`model id: qwen3:32b`,
  max context 32768 tokens)
- Freebase subset materialised as N-Triples domain subgraphs

> The service address is **not** hard-coded. Set it through the environment:
>
> ```bash
> export LLM_BASE_URL="http://localhost:11434/v1"
> export LLM_API_KEY="ollama"
> export LLM_MODEL="qwen3:32b"
> ```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the results

```bash
# 1) three main batches (four configurations: A/B/C/D)
python experiments/launch_experiment.py      # G1
python experiments/launch_experiment_g2.py   # G2
python experiments/launch_experiment_g3.py   # G3

# 2) hyper-parameter sensitivity (Table 7 / Table 8 in the paper)
#    Table 7 / Table 8 each have three columns:
#      sweep set (25 q) + holdout (25 q) + pooled (50 q)
python experiments/launch_E1_K_sweep.py        # K      -- sweep set
python experiments/launch_E1_K_extra.py        # K      -- holdout
python experiments/launch_E2_mindepth_sweep.py # d_min  -- sweep set
python experiments/launch_E2_d_extra.py        # d_min  -- holdout

# 3) the D' control experiment (separating the early-stopping contribution)
python experiments/d_prime_patch.py
```

Each run writes a JSON file under `results/` containing per-question records
(`id`, `success`, `time`, `hop`, `d_predict`, `early_stop`, `llm_calls`, ...).
The aggregated tables in the paper are computed from these records.

### Layout of `results/`

| Path | Contents |
|---|---|
| `G1_pog_4group.json`, `G2_…`, `G3_…` | the three main batches (Tables 2-6) |
| `E1_K_sweep/K_{5,10,15,20}/` | K sensitivity, sweep set (Table 7, col. 1) |
| `E1_K_extra/K_{5,10,15,20}/` | K sensitivity, holdout (Table 7, col. 2) |
| `E2_mindepth_sweep/d_{1,2,3}/` | d_min sensitivity, sweep set (Table 8, col. 1) |
| `E2_d_extra/d_{1,2,3}/` | d_min sensitivity, holdout (Table 8, col. 2) |
| `window2/pog_4group_20260803_*.json` | second time window, 3 batches (Table 9, col. 2) |

For Tables 7 and 8 the pooled column is obtained by simply concatenating the
per-question records of the two halves and recomputing the rate; the reported
latency is the mean over all 50 questions of the two halves.

Table 9 compares two time windows. Time window 1 is the main experiment
(`G1_pog_4group.json` … `G3_pog_4group.json`); time window 2 is the re-run
carried out later under a heavier server load, three batches stored under
`results/window2/`. In Table 9 the per-call latency of a group is the group's
mean per-question latency divided by its mean number of LLM calls per question,
so it is an approximation rather than a direct per-request measurement.

## Measurement conventions (important for reproducing the numbers)

We documented these explicitly in the paper because they change the reported values:

- **Quantiles** (P50/P90) use the `higher` definition, i.e.
  `numpy.percentile(x, q, method="higher")`. With numpy's default linear
  interpolation, the P90 of the baseline differs by about 10 s.
- **McNemar tests** are the **exact** (binomial) form, not the asymptotic
  chi-square form.
- **Confidence intervals** come from a **paired bootstrap** over questions,
  reported with the percentile method. The lower bound of the pooled interval
  is sensitive to the bootstrap seed: the difference can only take values of
  `k/300`, and the fraction of exactly-zero resamples (~0.9%) sits close to the
  2.5% quantile, so the bound alternates between 0.00 and 0.33.

## Data

The domain subgraphs (`military_kg_group1/2/3.nt`, about 46 MB each) and the
derived artefacts are archived separately (see the Data Availability statement
in the paper). The question set is derived from **ComplexWebQuestions**
(Talmor & Berant, NAACL 2018); please cite the original paper if you reuse it.

## Citation

If you use this code, please cite the paper (full reference to be added upon
publication).

## License

MIT — see `LICENSE`.

## Regenerating the figures

```bash
python make_figures_en.py
```

The script writes into a `figures/` directory **next to itself** (it uses no
absolute path), and saves each figure under **two names**: an internal name
(`figN_*.png`) and the paper's own numbering (`FigN_*.png`). The mapping is

| internal name | paper number |
|---|---|
| `fig1_accuracy.png` | Fig. 2 |
| `fig2_phase.png` | Fig. 3 |
| `fig3_sensitivity.png` | Fig. 4 |
| `fig4_framework.png` | Fig. 1 |

The `figures/` directory in this repository holds the paper-numbered copies.
It reads only `figdata/figdata.json`, which holds the values extracted from the result files listed above; the script itself needs nothing but matplotlib.

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Notes:

- `rdflib` is imported at module level by `src/freebase_func.py` and
  `experiments/build_high_match_dataset.py`, so it is required.
- `sentence-transformers` is used for the fuzzy-selection route. In
  `src/pog_core.py` the import is wrapped in `try/except` and falls back to
  keyword matching when the library is unavailable, but
  `experiments/launch_experiment.py` imports it at module level, so the main
  experiment needs it installed. Loading the `all-MiniLM-L6-v2` model also
  requires network access on first use.
- `urllib3` is imported directly by `src/llm.py` (to silence the
  insecure-request warning when TLS verification is disabled for the internal
  endpoint). It normally arrives with `requests`, but it is listed explicitly
  because the code references it by name.
- `matplotlib` is needed only by `make_figures_en.py`.
