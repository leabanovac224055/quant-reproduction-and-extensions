# Reproducing QUANT

A reproduction of QUANT (Dempster, Schmidt and Webb, *QUANT: a minimalist
interval method for time series classification*, DMKD 38:2377–2402, 2024),
a re-analysis of two of its claims, and attempts to extend it.

Course project for *AI for Time Series*, University College Dublin.

Lea Banovac and Donna Saji Sebastian.

## What we found

| | |
|---|---|
| The released code reproduces its published results on 138 of 142 datasets | mean absolute difference 0.0022 per dataset; over all 142 our run gives 0.8550, the value in the paper's own Fig. 6 |
| The four exceptions are the four datasets a published Correction to the bake-off revised | their published results do not match a run on the corrected data |
| Correcting for the comparisons the paper runs, its one significant result against the strongest interval methods no longer clears 0.05 | rSTSF, 0.0277 raw, 0.0553 under Holm |
| QUANT's lead over DrCIF is not general across domains | at parity on image, sensor and spectro (60 of 112 datasets); no domain with five or more datasets differs significantly |
| Reproduced under the paper's own protocol, the depth effect of its Fig. 8 is neither reproduced nor excluded | the 25-configuration surface spans 0.0059; paired 95 % intervals are about ±0.002 |
| Importance-based feature selection at 12.5 % of the feature space has no detectable accuracy cost | 95 % intervals within ±0.006; 2.1x to 7.8x less single-core classifier time, depending on dataset size |
| The interval-weighting extension the paper proposes does not help | no variant improves on the baseline, and no difference survives correction |

## Results against the paper

| | paper | ours |
|---|---|---|
| Mean accuracy, 142 datasets, their code | 0.8568 (released CSV), 0.8550 (Fig. 6) | 0.8550 |
| Mean accuracy, 138 datasets (Correction four excluded) | 0.8546 | 0.8546 |
| Mean accuracy, 112 datasets, our reimplementation | 0.8671 | 0.8670 |
| Wilcoxon p, QUANT vs rSTSF / DrCIF (Fig. 1) | 0.0277 / 0.1442 | 0.0277 / 0.1442 (`zero_method="pratt"`) |
| rSTSF after Holm correction | not reported | 0.0553 |
| Depth 6 → 8 at m/8 and m/16 (Fig. 8) | decline of about 0.002 | +0.0003 / +0.0001, 95 % CI about ±0.002 |

## Layout

```
src/                      experiment and verification scripts
  superseded/             earlier versions of experiments, not reported
notebooks/                analysis and figure generation
results/                  every result file the report draws on
  superseded/             results of the superseded runs
figures/                  the figures in the report, as PDF and PNG
exploratory_extensions/   two further extensions, described below
```

## Setup

This repository does not redistribute the authors' code or the data, because
they are not ours. Four downloads are needed.

**The QUANT implementation** (GPL-3.0), from
<https://github.com/angus924/quant>. Extract it so that the code sits at
`Quant/quant-main/quant-main/code/quant.py`. It also contains the published
accuracies, `results/accuracy_resamples_ucr142.csv`, which most comparisons
read.

**The 112 original UCR datasets**: `Univariate2018_ts.zip` from
<https://timeseriesclassification.com/dataset.php>. Extract it so that each
dataset folder sits at `data/Univariate2018_ts/Univariate_ts/<Dataset>/`.

**The 30 datasets added by the bake-off redux**: `30NewDatasets.zip` from
<https://drive.google.com/file/d/1vuh6mgNrNKjHr9MMRQP0J0_gGA4dE7E3/view>.
Extract them into the same `Univariate_ts/` folder. They keep their
underscore names (`AconityMINIPrinterLarge_eq` and so on);
`results/ucr142_local.txt` is the dataset list we used, since the list
shipped with the QUANT repository uses camel-case names the archive does not.

**The resample indices**: `JavaResampleIndices.zip` from
<https://drive.google.com/file/d/1FsG7Fp74y_TpaPhJ7U066ot8A07BPhr3/view>
(the tsml-eval repository also documents how they are generated). Extract it
so that the files sit at
`data/JavaResampleIndices/JavaResampleIndices/<Dataset>/resample<i>Indices_TRAIN.txt`.

Then, with Python 3.10:

```
pip install -r requirements.txt
```

`requirements-freeze.txt` records the exact environment the reported results
were produced in.

Every script resolves its paths from the repository root, so with those
directories in place nothing needs configuring. Set `QUANT_ROOT` if the data
and the QUANT code live elsewhere.

If the setup is correct, `python src/reproduce_fig8.py --list` reports exactly
100 eligible datasets, matching the paper, and `python src/check_indices.py`
reports no missing or invalid index files.

## Reproducing the results

Timings are from a 13th-generation Intel Core i9-13900H laptop with 16 GB of
RAM.

**Their code on all 142 datasets.** `run.py` is run unmodified, with our
dataset list and the Java resample indices. It writes `results.csv` only when
every dataset has finished, so the output folder must exist beforehand:

```powershell
cd Quant/quant-main/quant-main/code
mkdir ../../../../results/theirs142
python run.py -d ../../../../results/ucr142_local.txt -i ../../../../data/Univariate2018_ts/Univariate_ts -p ../../../../data/JavaResampleIndices/JavaResampleIndices -o ../../../../results/theirs142
cd ../../../..
mv results/theirs142/results.csv results/theirs142_results.csv
```

`run.py` defaults to 30 resamples, depth 6, `m/4` quantiles, 200 trees and
`max_features = 0.1`, the paper's settings, and uses every core.

**Everything else:**

```
# our reimplementation on the 112 equal-length datasets, then the comparison
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

# per-domain breakdown
python src/build_types.py                      # domain labels
python src/domain_analysis.py                  # QUANT by domain
python src/domain_vs_baselines.py              # QUANT vs other interval methods by domain
python src/raw_type_breakdown.py               # the same on aeon's raw labels
```

Each long-running script appends to its output file after every dataset and
resumes if interrupted. Add `--summarise` (or `--summarise-only` for
`reproduce_fig8.py`) to print the tables from results already on disk
without running anything.

The verification scripts are quick and need no long runs:

```
python src/check_assumptions.py           # resampling and the depth lever
python src/check_provenance_columns.py    # the feature provenance mapping
python src/check_fold0.py                 # fold 0 against the published results
python src/check_indices.py               # resample indices fit the data
python src/check_fig8_consistency.py      # the two protocols against each other
```

`check_fold0.py` covers six of the original datasets. The fold-0 figures the
report gives for Tools, Colposcopy, SharePriceIncrease and the four Correction
datasets come instead from resample 0 of `results/theirs142_results.csv`,
compared against the first column of the published
`accuracy_resamples_ucr142.csv`.

## Notebooks

| notebook | what it does |
|---|---|
| `report_figures.ipynb` | regenerates every figure in the report from `results/` |
| `fig8_analysis.ipynb` | the sensitivity analysis: is the depth effect there? |
| `domain_sensitivity.ipynb` | is QUANT's advantage uniform across domains, and how robust is it? |

`report_figures.ipynb` needs the published accuracies from the QUANT
repository; the other two read only files in `results/`.

## Further extensions

`exploratory_extensions/` holds two exploratory extensions by Donna Saji Sebastian, each with
its own README and requirements. Both add features to QUANT's own
representation rather than replacing it:

- **Temporal Feature-Enhanced QUANT** adds order-aware descriptors over the
  QUANT intervals, alongside the quantiles.
- **Change-Point Guided QUANT** adds quantiles over segments found by
  change-point detection.

Both are preliminary pilots, evaluated on eight datasets at resample 0; see
the report for what they show.

## Superseded runs

`src/superseded/` and `results/superseded/` hold earlier versions of two
experiments, kept for transparency: in one the dataset list was changed after
seeing a result, in another the feature subsets were drawn in a way that did
not realise the weights they were meant to. Nothing reported in the paper
comes from them. Each script carries a header explaining what replaced it.

## Note on the authors' code

`quant.py` and `run.py` are not modified anywhere in this repository. Where a
reimplementation was needed, it is a separate file
(`src/run_reproduction.py` reimplements the experiment;
`src/quant_provenance.py` reconstructs the feature layout;
`src/interval_weighting.py` wraps the classifier stage).