# Seed-repeat experiment checkpoints (Section 5.8)

Nine checkpoint files, one per `(batch, seed)` combination:

    G1_seed11  G1_seed22  G1_seed33
    G2_seed11  G2_seed22  G2_seed33
    G3_seed11  G3_seed22  G3_seed33

Each file holds the per-question records for two configurations, run on the
same question set with only the sampling seed of the inference stage changed:

  * `B_仅关系筛选` — PoG with relation pre-filtering only
  * `D_双模块全开` — PoG with relation pre-filtering **and** sufficiency judgment

Settings: `K = 10`, `d_min = 2`, `max_depth = 3`,
`POG_SAMPLE_SIZE = 12`, `POG_SAMPLE_STRATEGY = "balanced_by_compositionality"`.
Questions are taken in a fixed order after stratification by compositionality,
so different seeds process exactly the same question set — which is what makes
the paired per-question comparison in Section 5.8 valid.

## Three questions were excluded

Three questions hit an implementation defect in the fallback branch of the
relation filter: the branch referenced an undefined variable (`working` instead
of `all_relations`) and therefore raised `NameError` whenever the LLM-based
relation filter failed. One question in each of `G1_seed22`, `G2_seed22` and
`G3_seed22` was affected.

These questions are **excluded from the paired comparison** rather than counted
as misses, because the failure came from the harness, not from the method.
The defect is fixed in `src/pog_improvements.py` in this package; it did not
affect the main experiment (all 84 main-experiment files have zero occurrences).

## Reproducing

The checkpoints are the raw output of the run. The derived numbers reported in
Section 5.8 (9 pairs; D better in 1, B better in 5, 3 tied; mean difference
-3.87 percentage points; 95% bootstrap interval [-9.52, +1.90]) are computed
from these files by `tools/extract_seed_fields.py`.
