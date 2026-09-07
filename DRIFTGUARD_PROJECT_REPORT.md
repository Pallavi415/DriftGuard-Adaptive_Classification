# DriftGuard — Complete Project Findings & Key Insights

> **Official Project:** DriftGuard — Adaptive Classification Under Concept Drift  
> **Repository:** `DriftGuard-Adaptive-Classification`  
> **Core workflow:** **Learn → Detect → Adapt**

---

# 1. Project Overview

DriftGuard is a streaming classification project designed to study how classification models behave when the underlying data distribution changes over time.

The project does not treat the problem as a simple static classification task. Instead, the data is processed as a **sequential stream**, models learn incrementally, prediction errors are monitored for changes, and the models can adapt when a change is detected.

The central idea is:

**Learn → Detect → Adapt**

- **Learn:** incrementally learn from incoming observations.
- **Detect:** monitor prediction-error behavior using Page-Hinkley.
- **Adapt:** reduce the influence of stale information when a candidate change is detected.

The project compares this adaptive approach against non-adaptive streaming baselines.

---

# 2. Dataset

**Dataset:** Electricity / ELEC2 (`elecNormNew.arff`)

## Dataset characteristics

- Observations: **45,312**
- Original columns: **9**
- Predictors: **8**
- Target: `class`

Target distribution:

| Class | Count | Percentage |
|---|---:|---:|
| DOWN | 26,075 | 57.54% |
| UP | 19,237 | 42.46% |

Data quality findings:

- Missing values: **None**
- Duplicate rows: **None**
- Numerical predictors were already normalized to approximately **0–1**
- `day` behaves as a categorical feature
- Original sequential order was preserved

## Preprocessing

- `day` was one-hot encoded.
- `class` was mapped:
  - `DOWN → 0`
  - `UP → 1`
- `date` and `period` were retained as predictors at the modeling stage.
- The stream was **not randomly shuffled**.

After preprocessing:

**Model-ready dataset = 45,312 observations × 15 columns**

The dataset does **not** provide explicit ground-truth concept-drift labels.

---

# 3. Streaming Design

The original dataset order was preserved and divided into sequential batches.

## Configuration

- Batch size: **500**
- Total batches: **91**
- Full batches: **90**
- Final batch: **312 observations**

The batch size represents how data arrives in our experiment. It is not a claim that the original dataset naturally arrives in batches of 500.

## Evaluation protocol

Batch 1 is used for model warm-up.

Batches 2–91 follow:

**Predict → Evaluate → Update**

This is a form of **prequential evaluation**.

The current batch is predicted before its labels are used for model updating, which avoids current-batch label leakage.

Total evaluated observations:

**44,812**

Total evaluated batches:

**90**

---

# 4. Non-Adaptive Baselines

The baseline models establish what happens when the models learn continuously without explicit drift-triggered adaptation.

## 4.1 Incremental Gaussian Naive Bayes

The incremental Gaussian NB implementation maintains running statistics using **Welford's algorithm**.

Welford's algorithm allows mean and variance to be updated incrementally without repeatedly storing the entire historical dataset.

### Streaming performance

| Metric | Score |
|---|---:|
| Accuracy | **0.7141** |
| F1-score | **0.5116** |
| Recall | **0.4533** |

---

## 4.2 Windowed KNN

KNN uses a bounded recent-history window.

Initial configuration:

- Window size: **2,000 observations**
- `n_neighbors = 5`

The 2,000-observation window represents four 500-observation batches.

It was selected as an **experimental baseline**, not as a proven optimal value.

### Streaming performance

| Metric | Score |
|---|---:|
| Accuracy | **0.6970** |
| F1-score | **0.6133** |
| Recall | **0.5963** |

---

# 5. Baseline Insight

The two models have different strengths.

### Gaussian NB

- Higher overall accuracy
- Lower F1 and recall

### KNN

- Lower accuracy
- Higher F1 and recall

Therefore:

> **Neither baseline model is universally best across all metrics.**

This is important because the final model choice should depend on the objective rather than accuracy alone.

---

# 6. Page-Hinkley Drift Detection

Page-Hinkley was implemented from first principles in:

`src/driftaware/drift/page_hinkley.py`

The detector was also unit tested.

The purpose of Page-Hinkley in DriftGuard is to monitor **changes in prediction-error behavior**.

A prediction error is:

- `0` → correct prediction
- `1` → incorrect prediction

However, directly monitoring every individual binary error did not produce a useful detection signal.

---

# 7. Important Detection Experiment

## Individual prediction-error signal

Page-Hinkley was first applied to every individual prediction error.

Results:

- Gaussian NB: **39,523 detections**
- KNN: **40,833 detections**

The detections were clearly excessive and frequently occurred in consecutive observations.

### Decision

This approach was **rejected** as the primary detection signal.

The important lesson was:

> Monitoring every binary error made the detector too sensitive and produced an uninterpretable number of alerts.

---

# 8. Batch-Level Error Signal

Instead of monitoring individual errors, prediction errors were aggregated within each evaluated batch.

Each batch therefore produced one **error rate**.

The resulting signal contained:

- 90 evaluated batches
- Minimum error rate: **0.126**
- Maximum error rate: **0.624**

This produced a smoother and more interpretable signal for Page-Hinkley.

A figure was saved as:

`reports/figures/batch_error_rate.png`

---

# 9. Page-Hinkley Sensitivity — Gaussian NB

Different thresholds were tested to understand detector behavior.

| Threshold | Detections |
|---:|---:|
| 0.5 | 42 |
| 1.0 | 18 |
| **1.5** | **9** |
| 2.0 | 0 |
| 2.5 | 0 |

The configuration selected for adaptive Gaussian NB was:

- `delta = 0.01`
- `threshold = 1.5`

The choice was based on detection behavior rather than optimizing classification accuracy.

The resulting detections are considered **candidate change points**, not confirmed ground-truth drift events.

---

# 10. Page-Hinkley Sensitivity — KNN

For the KNN error signal:

| Threshold | Detections | Candidate batch |
|---:|---:|---:|
| 0.50 | 1 | 51 |
| 0.75 | 1 | 63 |
| **1.00** | **1** | **69** |
| 1.25 | 0 | — |
| 1.50 | 0 | — |

The configuration selected for adaptive KNN was:

- `delta = 0.01`
- `threshold = 1.0`

This produced one candidate change point around **Batch 69**.

Again, Batch 69 is a **detector output**, not a ground-truth drift label.

---

# 11. Adaptive Gaussian NB

The adaptive Gaussian NB extends the incremental Gaussian NB model.

When Page-Hinkley detects a candidate change:

> The influence of older Gaussian statistics is reduced.

Configuration:

- Decay factor: **0.5**
- Page-Hinkley delta: **0.01**
- Page-Hinkley threshold: **1.5**
- Adaptation events: **1**

## Results

| Metric | Non-Adaptive | Adaptive | Change |
|---|---:|---:|---:|
| Accuracy | 0.7141 | **0.7162** | **+0.0021** |
| F1-score | 0.5116 | **0.5161** | **+0.0045** |
| Recall | 0.4533 | **0.4595** | **+0.0062** |

### Interpretation

Adaptive NB improved across all three reported metrics.

The improvement is real within this experiment, but it is **modest**.

The appropriate conclusion is not that adaptive NB is universally superior. Instead:

> **Explicitly reducing the influence of older statistics provided a small performance benefit for Gaussian NB in this streaming experiment.**

---

# 12. Adaptive Windowed KNN

The adaptive KNN normally retains the latest 2,000 observations.

When Page-Hinkley detects a candidate change:

> The current window is reduced to approximately half, allowing recent observations to become dominant more quickly.

Configuration:

- Normal window: **2,000**
- `n_neighbors = 5`
- Page-Hinkley delta: **0.01**
- Page-Hinkley threshold: **1.0**
- Adaptation events: **1**

## Results

| Metric | Non-Adaptive | Adaptive | Change |
|---|---:|---:|---:|
| Accuracy | 0.6970 | **0.6972** | **+0.0002** |
| F1-score | 0.6133 | **0.6140** | **+0.0007** |
| Recall | 0.5963 | **0.5970** | **+0.0007** |

### Interpretation

Adaptive KNN improved slightly across all three metrics, but the differences were extremely small.

This is reasonable because KNN already uses a recent 2,000-observation window.

Therefore:

> **KNN already has a natural forgetting mechanism, so explicit drift-triggered forgetting produced only marginal additional benefit in this experiment.**

---

# 13. Final Model Comparison

| Model | Accuracy | F1-score | Recall |
|---|---:|---:|---:|
| Incremental Gaussian NB | 0.7141 | 0.5116 | 0.4533 |
| **Adaptive Gaussian NB** | **0.7162** | **0.5161** | **0.4595** |
| Windowed KNN | 0.6970 | 0.6133 | 0.5963 |
| **Adaptive Windowed KNN** | **0.6972** | **0.6140** | **0.5970** |

---

# 14. Best Model by Metric

## Accuracy

**Adaptive Gaussian NB — 0.7162**

It achieved the highest overall accuracy.

## F1-score

**Adaptive Windowed KNN — 0.6140**

It achieved the highest balance between precision and recall for the positive class.

## Recall

**Adaptive Windowed KNN — 0.5970**

It identified the positive class more effectively than Gaussian NB.

### Overall decision

There is **no universal winner**.

- If overall accuracy is the priority → **Adaptive Gaussian NB**
- If F1 or recall is more important → **Adaptive Windowed KNN**

---

# 15. Most Important Project Insight

The central comparison in DriftGuard is **not simply NB vs KNN**.

The key research question is:

> **Does explicitly detecting changes and adapting the model improve performance on a changing data stream?**

To answer this, each model was evaluated in two forms:

### Gaussian NB

**Incremental NB → Adaptive NB**

### KNN

**Windowed KNN → Adaptive KNN**

This makes the experiment an **adaptation study**, rather than just a model leaderboard.

---

# 16. What Adaptation Actually Achieved

### Gaussian NB

Adaptation provided a more noticeable improvement:

- Accuracy: **+0.0021**
- F1: **+0.0045**
- Recall: **+0.0062**

### KNN

Adaptation provided only marginal improvement:

- Accuracy: **+0.0002**
- F1: **+0.0007**
- Recall: **+0.0007**

### Overall conclusion

> **Adaptation helped both models slightly, but its benefit was much more noticeable for Gaussian NB than for KNN.**

This is consistent with the model designs:

- Gaussian NB accumulates historical statistics and therefore benefits from explicitly reducing stale information.
- Windowed KNN already forgets old observations through its bounded recent-data window.

---

# 17. What We Should NOT Claim

The results do **not** support the following claims:

- Adaptation always improves classification.
- Page-Hinkley found the true drift points.
- Batch 69 is definitely a real-world drift event.
- Adaptive KNN is substantially better than KNN.
- Adaptive NB is universally the best classifier.
- The selected thresholds are globally optimal.

The defensible statement is:

> **The adaptive strategy produced small improvements for both models under the selected experimental configuration, with a more noticeable benefit for Gaussian NB.**

---

# 18. Production Architecture

The project separates experimentation from reusable implementation.

```text
drift_aware_classification/
│
├── config/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_dataset_validation_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_streaming_simulation.ipynb
│   ├── 04_incremental_naive_bayes.ipynb
│   ├── 05_windowed_knn.ipynb
│   ├── 06_page_hinkley_drift_detection.ipynb
│   ├── 07_adaptive_models.ipynb
│   └── 08_final_evaluation.ipynb
│
├── reports/
│   ├── figures/
│   └── results/
│
├── scripts/
├── src/
│   └── driftaware/
│       ├── data/
│       ├── preprocessing/
│       ├── streaming/
│       ├── models/
│       ├── drift/
│       ├── evaluation/
│       └── utils/
│
├── tests/
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

The design principle is:

**Notebooks = experiment and explanation**  
**`src/` = reusable production logic**  
**`tests/` = verification**

---

# 19. Important Implementation Decisions

These decisions are locked for the project:

- Preserve original stream order.
- Do not randomly shuffle the stream.
- Use batch size **500**.
- Use prequential **Predict → Evaluate → Update** evaluation.
- Batch 1 is warm-up.
- Use Welford's algorithm for incremental Gaussian statistics.
- Use a bounded KNN window.
- Initial KNN window = **2,000**.
- Initial KNN neighbors = **5**.
- Detect changes through prediction-error behavior.
- Treat detector outputs as candidate change points.
- Do not claim exact natural drift locations without evidence.
- Keep adaptive and non-adaptive baselines comparable.
- Do not tune models solely to improve accuracy.
- Keep reusable code in `src/`.
- Test new production components.
- Save meaningful results and figures under `reports/`.
- Keep code readable and humanized.
- Use concise first-person notebook narrative.
- Follow **WHY → DO → FIND → DECIDE**.
- Avoid unnecessary abstraction or clever code.

---

# 20. Project Limitations

1. **No ground-truth drift labels**

   The dataset does not identify exact moments of concept drift. Therefore, Page-Hinkley detections are candidate change points.

2. **Detector parameters**

   Page-Hinkley parameters were selected by observing detector behavior on the error signal. They were not optimized against known drift labels.

3. **Single primary benchmark**

   The conclusions come from one non-stationary benchmark dataset. Additional datasets would provide stronger evidence.

4. **KNN window size**

   The 2,000-observation window is an experimental baseline, not a proven optimum.

5. **Modest adaptive improvements**

   The observed improvements are small, especially for KNN.

6. **Final metric aggregation**

   The current project comparisons use the established batch-level result summaries. Any future publication-grade evaluation should also retain and report observation-level aggregate metrics where appropriate.

---

# 21. Important Project Artifacts

## Figures

```text
reports/figures/
├── feature_class_comparison.png
├── feature_correlation.png
├── baseline_model_comparison.png
├── incremental_nb_streaming_performance.png
├── windowed_knn_streaming_performance.png
├── batch_error_rate.png
├── page_hinkley_candidate_change_points.png
├── adaptive_model_comparison.png
└── final_model_performance.png
```

## Results

```text
reports/results/
├── baseline_results.csv
├── streaming_configuration.csv
├── incremental_nb_streaming_results.csv
├── incremental_nb_streaming_summary.csv
├── windowed_knn_streaming_results.csv
├── windowed_knn_streaming_summary.csv
├── streaming_model_comparison.csv
├── adaptive_nb_streaming_results.csv
├── adaptive_nb_streaming_summary.csv
├── adaptive_knn_streaming_results.csv
├── adaptive_knn_streaming_summary.csv
├── adaptive_model_comparison.csv
└── final_model_comparison.csv
```

---

# 22. Notebook Progress

```text
01 Dataset Validation & EDA       ✓
02 Baseline Models                ✓
03 Streaming Simulation           ✓
04 Incremental Gaussian NB        ✓
05 Windowed KNN                   ✓
06 Page-Hinkley Drift Detection   ✓
07 Adaptive Models                ✓
08 Final Evaluation               ✓
```

---

# 23. Final Project Statement

DriftGuard demonstrates a complete experimental workflow for adaptive classification on a non-stationary data stream.

The project combines:

**Incremental Learning**
→ models learn sequentially from incoming data.

**Drift Detection**
→ Page-Hinkley monitors changes in batch-level prediction error.

**Model Adaptation**
→ detected changes trigger mechanisms that reduce the influence of stale information.

The final results show that adaptation can provide a measurable, although modest, improvement under the selected experimental configuration.

The strongest accuracy was achieved by **Adaptive Gaussian NB**, while **Adaptive Windowed KNN** achieved the strongest F1-score and recall.

The main technical insight is that the usefulness of adaptation depends on the model's existing memory mechanism. Gaussian NB benefits more from explicit forgetting, whereas KNN already forgets through its sliding window.

The project therefore supports the feasibility of the **Learn → Detect → Adapt** architecture while also demonstrating why drift detection and adaptation must be evaluated carefully rather than assumed to improve performance.

---

# 24. Current Status

The experimental phase is complete.

Next project phase:

**Repository cleanup → production-code review → full test verification → documentation → final GitHub checkpoint**

This document should be treated as the consolidated reference for the project's methodology, locked decisions, results, key insights, and limitations.
