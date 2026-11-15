# Reproducing QUANT

A reproduction of QUANT (Dempster, Schmidt and Webb, *QUANT: a minimalist
interval method for time series classification*, DMKD 38:2377–2402, 2024),
a re-analysis of two of its claims, and attempts to extend it.

Course project for *AI for Time Series*, University College Dublin.

Lea Banovac and Donna Saji Sebastian.

## What we found

| | |
|---|---|
| The released code reproduces its published results on 138 of 142 datasets | mean absolute difference 0.0022 per dataset |
| The four exceptions are the four datasets a published Correction to the bake-off revised | their published results do not match a run on the corrected data |
| Correcting for the comparisons the paper runs, its one significant result against the strongest interval methods no longer clears 0.05 | rSTSF, 0.0277 raw, 0.0553 under Holm |
| Reproduced under the paper's own protocol, the depth effect of its Fig. 8 is not detectable where it is claimed | the whole 25-configuration surface spans 0.0059 |
| Importance-based feature selection keeps full-model accuracy at 12.5 % of the feature space | 2.1x to 7.8x less single-core classifier time, depending on dataset size |
| The interval-weighting extension the paper proposes does not help | weighting by interval length is reliably worse |

## Layout

```
src/              experiment and verification scripts
  superseded/     earlier versions of experiments, not reported
notebooks/        analysis and figure generation
results/          every result file the report draws on
  superseded/     results of the superseded runs
figures/          the figures in the report, as PDF and PNG
Donna/            two further extensions, described below
```

## Setup

Two things this repository does not redistribute, because they are not ours:

**The QUANT implementation**, which is GPL-3.0 licensed. Download it from
<https://github.com/angus924/quant> and extract it so that the code sits at
`Quant/quant-main/quant-main/code/quant.py`.

**The UCR archive**, from <https://timeseriesclassification.com>. Extract the
univariate `.ts` files so that each dataset folder sits at
`data/Univariate2018_ts/Univariate_ts/<Dataset>/`. The 30 datasets added by
the bake-off redux keep their underscore names
(`AconityMINIPrinterLarge_eq` and so on); `results/ucr142_local.txt` is the
dataset list we used, since the list shipped with the QUANT repository uses
camel-case names the archive does not.

Then:

```
pip install -r requirements.txt
```

`requirements-freeze.txt` records the exact environment the reported results
were produced in.

Every script resolves its paths from the repository root, so with those two
directories in place nothing needs configuring. Set `QUANT_ROOT` if the data
and the QUANT code live elsewhere.

If the setup is correct, `python src/reproduce_fig8.py --list` reports exactly
100 eligible datasets, matching the paper.

## Reproducing the results

Timings are from a 13th-generation Intel Core i9-13900H laptop with 16 GB of
RAM.

```
# the reproduction: their code on all 142 datasets, then ours on 112
python src/run_reproduction.py                 # ~4 h, multi-core
python src/compare_three_way.py                # seconds

# the sensitivity analysis under the paper's own protocol
python src/reproduce_fig8.py --list            # the eligible pool
python src/reproduce_fig8.py                   # ~18 h, multi-core

# feature selection, the one improvement that works
python src/run_reduction_v2.py --list          # the three dataset groups
python src/run_reduction_v2.py --group random  # ~5 h, single-core
python src/run_reduction_v2.py --group small
python src/run_reduction_v2.py --group large
python src/run_reduction_f1.py                 # class-balanced metrics

# interval weighting, the extension that does not
python src/run_weighting_v2.py                 # ~11 h, multi-core
```

Each appends to its output file after every dataset and resumes if
interrupted. Add `--summarise` (or `--summarise-only` for `reproduce_fig8.py`)
to print the tables from results already on disk without running anything.

The verification scripts are quick and need no long runs:

```
python src/check_assumptions.py           # resampling and the depth lever
python src/check_provenance_columns.py    # the feature provenance mapping
python src/check_fold0.py                 # fold 0 against the published results
python src/check_fig8_consistency.py      # the two protocols against each other
```

## Notebooks

| notebook | what it does |
|---|---|
| `report_figures.ipynb` | regenerates every figure in the report from `results/` |
| `fig8_analysis.ipynb` | the sensitivity analysis: is the depth effect there? |
| `domain_sensitivity.ipynb` | is QUANT's advantage uniform across domains, and how robust is it? |

`report_figures.ipynb` needs the published accuracies from the QUANT
repository; the other two read only files in `results/`.

## Further extensions

`Donna/` holds two more attempts to improve the method, each with its own
README and requirements. Both add features to QUANT's own representation
rather than replacing it:

- **Temporal Feature-Enhanced QUANT** adds order-aware descriptors over the
  same intervals, alongside the quantiles.
- **Change-Point Guided QUANT** adds quantiles over segments found by
  change-point detection.

## Superseded runs

`src/superseded/` and `results/superseded/` hold earlier versions of two
experiments. They are kept because the report says they were discarded and
why: in one the dataset list was changed after seeing a result, in another
the feature subsets were drawn in a way that did not realise the weights they
were meant to. Nothing reported in the paper comes from them. Each script
carries a header explaining what replaced it.

## Note on the authors' code

`quant.py` is not modified anywhere in this repository. Where a
reimplementation was needed, it is a separate file
(`src/quant_provenance.py` reconstructs the feature layout;
`src/interval_weighting.py` wraps the classifier stage).