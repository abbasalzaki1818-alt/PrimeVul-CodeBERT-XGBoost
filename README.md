# Explainable CodeBERT–XGBoost for Realistic Source-Code Vulnerability Detection

This repository contains the implementation used in the study:

**Explainable CodeBERT–XGBoost for Realistic Source-Code Vulnerability Detection under Severe Class Imbalance**

The framework uses frozen CodeBERT representations as semantic features and XGBoost as the downstream classifier. Model development is performed under the original PrimeVul train/validation/test protocol, with validation-only hyperparameter, class-weight, and threshold selection.

## Overview

Pipeline:

1. PrimeVul data inspection
2. Data preprocessing
3. Frozen CodeBERT embedding extraction
4. XGBoost training and validation-guided model selection
5. Baseline training
6. Explainability analysis
7. Statistical validation
8. Standalone PrimeVul paired evaluation

The held-out test set is used only for final evaluation.

## Dataset

This project uses the **PrimeVul Original Release** introduced by Ding et al.

Official repository:

https://github.com/DLVulDet/PrimeVul

The official PrimeVul repository notes that the newer `v0.1` release contains additional metadata but fewer samples. The experiments in this repository use the **Original Release**, corresponding to the complete sample set used in the PrimeVul paper.

Expected files inside a local `dataset/` directory:

```text
dataset/
├── primevul\\\_train.jsonl
├── primevul\\\_valid.jsonl
├── primevul\\\_test.jsonl
├── primevul\\\_train\\\_paired.jsonl
├── primevul\\\_valid\\\_paired.jsonl
└── primevul\\\_test\\\_paired.jsonl
```

Dataset files are intentionally excluded from this repository. Please obtain them from the official PrimeVul source.

## Dataset Statistics

|Split|Vulnerable|Benign|Total|
|-|-:|-:|-:|
|Train|5,574|178,853|184,427|
|Validation|699|24,731|25,430|
|Test|695|25,216|25,911|

Paired test subset: **564 vulnerable–benign pairs (1,128 functions)**.

## Project Structure

```text
PrimeVul\\\_CodeBERT\\\_XGBoost/
├── main.py
├── paired\\\_evaluation.py
├── check\\\_all\\\_paired.py
├── check\\\_paired\\\_structure.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── data\\\_inspection.py
│   ├── preprocessing.py
│   ├── embedding\\\_extractor.py
│   ├── train\\\_xgboost.py
│   ├── train\\\_baselines.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── statistical\\\_validation.py
│   └── utils.py
├── dataset/      # local only; excluded from Git
└── outputs/      # generated outputs; excluded from Git
```

## Environment

Experiments were run with:

* Python 3.11
* PyTorch 2.11.0 + CUDA 12.8
* Transformers 5.15.0
* XGBoost 3.2.0
* scikit-learn 1.9.0
* imbalanced-learn 0.14.2
* SHAP 0.51.0
* NVIDIA GeForce RTX 3080 Laptop GPU (16 GB VRAM)

Install dependencies with:

```bash
pip install -r requirements.txt
```

If your system does not support CUDA 12.8, install the PyTorch build appropriate for your hardware before running the embedding stage.

## Running the Pipeline

The main pipeline supports the following stages:

```bash
python main.py --stage inspect
python main.py --stage preprocess
python main.py --stage embed
python main.py --stage train\\\_xgboost
python main.py --stage train\\\_baselines
python main.py --stage explain
python main.py --stage statistical\\\_validation
```

To run all main stages:

```bash
python main.py --stage all
```

The PrimeVul paired evaluation is currently implemented as a standalone script:

```bash
python paired\\\_evaluation.py
```

## Representation

Source-code functions are encoded using:

```text
microsoft/codebert-base
```

Configuration:

* Frozen CodeBERT encoder
* Maximum sequence length: 512 tokens
* Attention-mask-aware mean pooling
* Embedding dimension: 768

Observed truncation rates:

* Train: 25.59%
* Validation: 25.58%
* Test: 26.32%

## Final XGBoost Configuration

Validation-selected settings:

|Parameter|Final value|
|-|-:|
|`n\\\_estimators`|400|
|`max\\\_depth`|8|
|`learning\\\_rate`|0.05|
|`min\\\_child\\\_weight`|1|
|`subsample`|0.8|
|`colsample\\\_bytree`|0.8|
|`tree\\\_method`|`hist`|
|`eval\\\_metric`|`aucpr`|
|`scale\\\_pos\\\_weight`|1.0|
|Main F1 threshold|0.0851544887|
|Recall-oriented F2 threshold|\~0.0416|
|Random seed|42|

The final class-weight search selected `scale\\\_pos\\\_weight = 1.0`; therefore, the proposed model should not be described as a class-weighted XGBoost model.

## Main Test Results

At the validation-selected F1 threshold:

|Metric|Result|
|-|-:|
|Accuracy|0.9470|
|Balanced Accuracy|0.6160|
|Precision|0.1765|
|Recall|0.2662|
|F1|0.2123|
|F2|0.2416|
|Specificity|0.9658|
|ROC-AUC|0.8155|
|PR-AUC|0.1567|
|MCC|0.1902|

Confusion matrix:

* TP = 185
* FN = 510
* FP = 863
* TN = 24,353

Low-FPR evaluation:

* VD-S @ 0.5% FPR = 0.8993
* VD-S @ 1% FPR = 0.8504

## Baselines

All downstream baselines use the same frozen CodeBERT embeddings.

|Model|F1|ROC-AUC|
|-|-:|-:|
|Logistic Regression|0.1676|0.8082|
|Linear SVM|0.1641|0.8078|
|Random Forest|0.1740|0.8046|

SMOTE ablation:

* F1 = 0.1780
* PR-AUC = 0.1253
* ROC-AUC = 0.8017

## Statistical Validation

A stratified bootstrap with 2,000 resamples was used while preserving the original class counts.

Selected 95% confidence intervals:

* Precision: \[0.1567, 0.1975]
* Recall: \[0.2345, 0.2993]
* F1: \[0.1887, 0.2369]
* F2: \[0.2142, 0.2710]
* ROC-AUC: \[0.8003, 0.8302]
* PR-AUC: \[0.1348, 0.1847]
* MCC: \[0.1653, 0.2168]

McNemar comparison against Random Forest:

* Statistic: 174.5987
* p-value: 7.326e-40

The McNemar result indicates a significant difference in prediction-correctness patterns and should not be interpreted by itself as proof of practical superiority.

## Explainability

The explainability stage includes:

* Global SHAP analysis over CodeBERT embedding dimensions
* Local prediction-aware explanations for:

  * 2 true positives
  * 1 false negative
  * 1 false positive
  * 1 true negative

SHAP values are interpreted as feature attribution over latent CodeBERT dimensions, not as direct causal vulnerability semantics.

## PrimeVul Paired Evaluation

The final fixed model was evaluated on 564 vulnerable–benign pairs using the same validation-selected threshold.

|Outcome|Pairs|Percentage|
|-|-:|-:|
|P-C|14|2.48%|
|P-V|127|22.52%|
|P-B|411|72.87%|
|P-R|12|2.13%|

These results are reported as a realistic stress test and are not used for model selection.

## Reproducibility Notes

* Original PrimeVul splits are preserved.
* The test set is not used to tune hyperparameters, class weights, or decision thresholds.
* CodeBERT embeddings are extracted separately for train, validation, and test.
* The paired evaluation uses the already-fixed final threshold.
* Dataset files, virtual environments, and generated large outputs are excluded from Git.

## Data and Code Availability

Dataset:

https://github.com/DLVulDet/PrimeVul

Code:

https://github.com/abbasalzaki1818-alt/PrimeVul-CodeBERT-XGBoost

## Citation

If you use PrimeVul, please cite the original dataset paper:

```bibtex
@inproceedings{ding2025primevul,
  title={Vulnerability Detection with Code Language Models: How Far Are We?},
  author={Ding, Yangruibo and Fu, Yanjun and Ibrahim, Omniyyah and Sitawarin, Chawin and Chen, Xinyun and Alomair, Basel and Wagner, David and Ray, Baishakhi and Chen, Yizheng},
  booktitle={2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE)},
  pages={1729--1741},
  year={2025},
  doi={10.1109/ICSE55347.2025.00038}
}
```

## Notes

This repository is intended to support reproducibility of the accompanying research. The reported results should be interpreted under the severe class imbalance and realistic evaluation constraints of PrimeVul.

