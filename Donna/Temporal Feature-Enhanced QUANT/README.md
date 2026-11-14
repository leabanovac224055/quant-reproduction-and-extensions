# Temporal Feature-Enhanced QUANT
An experimental extension of **QUANT** for time-series classification. This project augments QUANT's interval-quantile representation with temporal-order descriptors and evaluates the combined feature representation using an Extra Trees classifier.

## What this project does
1. Loads univariate time-series datasets in the UCR/UEA `.ts` format.
2. Applies the original QUANT interval-quantile transform.
3. Extracts additional temporal features from the same intervals.
4. Concatenates the QUANT and temporal feature vectors.
5. Trains an `ExtraTreesClassifier` and records classification accuracy and runtime.

The temporal feature implementation computes up to ten descriptors per interval:

- upward movement ratio
- downward movement ratio
- direction-change ratio
- mean increasing-run length (normalised)
- mean decreasing-run length (normalised)
- local peak ratio
- local valley ratio
- autocorrelation at lags 1, 2, and 3


## Requirements

- Python 3.10 or newer
- PyTorch
- NumPy
- pandas
- scikit-learn
- tqdm

## Setup
### 1. Clone the repository
### 2. Create and activate a virtual environment
### 3. Install dependencies
### 4. Run the temporal-feature experiment
### 5. Existing result files



## Result

The experimental evaluation was conducted on datasets to compare the original QUANT implementation with three temporal variations (Temporal 1, Temporal 2 and Temporal 3). The original QUANT achieved an average classification accuracy of 81.17%, while Temporal 1, Temporal 2 and Temporal 3 achieved average accuracies of 80.92%, 80.71% and 80.97%, respectively. Across all three temporal experiments, the combined average accuracy was 80.87%, representing a marginal decrease of approximately 0.30 percentage points compared with the original QUANT. The average execution time across the temporal experiments was approximately 12.68 seconds per dataset. Overall, these findings indicate that the temporal modifications provide mixed results and do not consistently outperform the original QUANT implementation. Further experimentation with different temporal configurations, multiple resampling runs and parameter settings is necessary to determine whether temporal information can improve the robustness and classification performance of QUANT.

## References

- Dempster, A., Schmidt, D. F., & Webb, G. I. (2024). *QUANT: A Minimalist Interval Method for Time Series Classification*. Data Mining and Knowledge Discovery. https://doi.org/10.1007/s10618-024-01036-9
- Original QUANT project: https://github.com/angus924/quant


