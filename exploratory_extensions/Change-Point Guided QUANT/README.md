# CP-QUANT: Change-Point-Guided QUANT for Time-Series Classification

This repository contains an experimental implementation of CP-QUANT alongside an original QUANT baseline. The project applies interval-based time-series feature extraction and evaluates classification performance using an Extra Trees classifier.

## Requirements

- Python 3.10 or later is recommended.
- NumPy
- pandas
- PyTorch
- scikit-learn
- tqdm
- ruptures

## Setup
### 1. Clone the repository
### 2. Create and activate a virtual environment
### 3. Install dependencies
### 4. Run the temporal-feature experiment
### 5. Existing result files

## Results

The experimental results demonstrate that the proposed Change Point-guided QUANT (CP-QUANT) approach achieves a slight improvement in classification accuracy compared to the original QUANT implementation across the evaluated datasets. The original QUANT model obtained an average accuracy of 81.27%, whereas CP-QUANT achieved an average accuracy of 81.86%, representing an improvement of approximately 0.59 percentage points. These results suggest that incorporating change-point guidance can provide modest improvements in classification performance across different time-series datasets. However, the results are based on a single resampling run (resample 0), and further experiments across multiple resampling runs are required to establish the consistency and statistical significance of these improvements.



## References

- Dempster, A., Schmidt, D. F., & Webb, G. I. **QUANT: A Minimalist Interval Method for Time Series Classification.** arXiv:2308.00928. https://arxiv.org/abs/2308.00928
- Original QUANT code referenced in the source: https://github.com/angus924/aaltd2024

