# Data processing

How the evaluation data of the paper was produced from public sources, and
which parts of that procedure are shipped in this repository.

## 1. Sources (both public)

| Source | Used for | Where to obtain |
|---|---|---|
| **ComplexWebQuestions (CWQ)** — Talmor and Berant, 2018 | the questions, together with their official SPARQL annotations | released by the original authors with the dataset |
| **Freebase** | the knowledge graph from which each batch's subgraph is extracted | publicly available from its original source |

No scraped, private or licensed data is involved.

## 2. Question selection

1. Read the official CWQ SPARQL annotations.
2. Keep the questions whose SPARQL contains a `military.*` relation prefix.
   This yields **341 military-domain questions**.
3. Of these, **300** are split into three batches of 100 questions each
   (**G1, G2, G3**) and used in the paper. The remaining 41 form a reserve
   analysis set and are not used for the reported results.
4. Within a batch, the question order is fixed. The seed-repeat experiment of
   Section 5.8 relies on this: because the sampling itself is deterministic
   (a balanced selection by question compositionality, then a fixed order),
   different sampling seeds of the inference stage process exactly the same
   question set, which is what makes the paired per-question comparison valid.

## 3. Subgraph construction

For each batch, a military-domain question-answering subgraph is extracted from
Freebase. As described in Section 5.1 of the paper, the construction involves

* **entity-level expansion** around the topic entities,
* **missing-value simulation**, and
* **noise injection**,

with the **gold answer entity used as an anchor**. The four configurations
A/B/C/D are evaluated on exactly the same questions, topic entities, subgraph,
and missing-value / noise settings, so that the comparison is controlled.

Because the gold answer entity is used as an anchor at this stage, the
evaluation is a *CWQ-derived closed-world subgraph* evaluation: it measures the
efficiency of path search on a fixed question-answering subgraph, not
end-to-end retrieval generalisation on CWQ without an answer prior.

## 4. What this repository ships

| Path | What it does | Relation to the steps above |
|---|---|---|
| `experiments/build_high_match_dataset.py` | samples from `cwq.json` with the configured size and strategy, scores each question's match against the local subgraph heuristically, labels them high / medium / low, and exports the high-match subset plus a statistics report | step 2 (question selection and match scoring) |
| `experiments/check_cwq.py` | prints the field structure of the CWQ file, to confirm field names before processing | step 1 (input inspection) |
| `src/freebase_func.py` | loads the Freebase RDF graph and provides one-hop neighbours, relations and entity names | the graph access layer used by step 3 |
| `configs/config.py` | `POG_SAMPLE_SIZE`, `POG_SAMPLE_STRATEGY`, `POG_MAX_DEPTH`, `POG_MIN_DEPTH_BEFORE_STOP`, and the dataset path resolution | the parameters of steps 2 and 3 |

> **Honest coverage note.** Step 3 — entity-level expansion, missing-value
> simulation and noise injection — was carried out in the experimental
> environment described in Section 5.1. This repository ships the question
> selection and match-scoring script and the graph access layer, but
> **not a standalone subgraph-construction script**. We state this explicitly
> rather than leave the reader to infer that the full pipeline is here.

## 5. Running the shipped part

```bash
pip install -r requirements.txt
export POG_DATASET_FILE=/path/to/cwq.json
export POG_SAMPLE_SIZE=12
export POG_SAMPLE_STRATEGY=balanced_by_compositionality
python experiments/build_high_match_dataset.py
```

Output goes to `experiment_results/` (JSON subset plus a statistics report).

## 6. Why the derived data is not deposited separately

The questions come from a public dataset and the knowledge graph from a public
source. The derived artifacts — the question subsets and the per-question
result files — are **included in this repository** under `results/`, and can
also be regenerated from the public sources by following Sections 2–3 above.
We therefore do not make a separate deposit of a "processed dataset".
