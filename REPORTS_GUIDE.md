# Reports Guide: Image Quality Assessment Using Deep Learning

This file is the single reference for the project and its four reports. Section 2 is the source of truth for every design decision; the code and all reports must agree with it.

There are exactly four reports:

| No. | Report | What it covers |
|---|---|---|
| 1 | Progress Report 1: Problem Study, Dataset and Model Design | The problem, background study, environment, dataset, split, preprocessing, model design, training and inference plan |
| 2 | Progress Report 2: Data Pipeline and Model Implementation | Dataset verification, split, preprocessing and tf.data pipeline, model code, sanity checks |
| 3 | Progress Report 3: Model Training and Validation | Two-phase training, validation results, label and diagnostic thresholds, problems fixed |
| 4 | Final Project Report | The complete project, including test results, the inference pipeline and the Streamlit application |

Progress Report 1 is written and stored in `reports/progress_report_1/`. Reports 2 and 3 are short progress reports. Report 4 is the full project report and must stand on its own.

---

## 1. How to use this guide

1. Finish the work that the report covers (see the notebook map in Section 3).
2. Collect every item in that report's **Evidence checklist** into `reports/<report_folder>/evidence/`.
3. Put the report's figures in `reports/<report_folder>/figures/`.
4. Write the report using **Section 4 (Format)**, **Section 5 (Style)** and its specification in **Section 7**.
5. Run **Section 8 (Final checks)** before submitting.

To generate a report with an AI assistant, use the prompt in **Section 9** and paste the real evidence. Never generate a report without evidence; the numbers will be invented.

---

## 2. Project facts (single source of truth)

### 2.1 Basic details

| Item | Value |
|---|---|
| Project title | Image Quality Assessment Using Deep Learning |
| Task | No-Reference Image Quality Assessment (NR-IQA), regression |
| Submitted to | Dr. Ratnadeep Dey *(confirm)* |
| Submitted by / Registration No. | *(fill in)* |
| Reference paper | Sang et al., "No-Reference Image Quality Assessment Leveraging GenAI Images," IEEE TIP, vol. 34, pp. 6204–6214, 2025 |
| Language / framework | Python, TensorFlow with Keras |
| Training environment | Google Colab (GPU runtime) |
| Demo environment | Local machine (CPU is enough), Streamlit |
| Main notebook | `notebooks/iqa_training_colab.ipynb` (self-contained; `src/` is refactored from it) |

**Scope limits:** no GenAI image generation, cold diffusion, Vision Transformer, MCAB/SSAM attention, self-supervised pre-training or model ensembles. The reference paper is used for problem framing and evaluation only.

### 2.2 Dataset: KonIQ-10k

| Item | Value |
|---|---|
| Images | 10,073 real-world photographs (Flickr, YFCC100M) |
| Distortions | Authentic, not synthetic |
| Ratings | ~1.2 million crowd ratings, ~120 per image |
| Rating scale | Individual ratings are 5-point (bad to excellent). The published MOS is rescaled to **0-100**, and the values actually present run from 3.91 to 88.39. All work in this project uses the 0-100 scale. |
| MOS distribution | Concentrated in the middle; few very poor or excellent images |
| Resolution | 1024 × 768 originals; **512 × 384 version used** |
| Image archive | `koniq10k_512x384.zip` (official KonIQ-10k download) |
| Scores file | `koniq10k_distributions_sets.csv` (subpic/koniq repository, `metadata/`) or `koniq10k_scores_and_distributions.csv` (official archive) |
| Local location | `data/raw/` (never committed) |

Column names must be checked in the notebook before use (expected: `image_name`, `MOS`, and a set column in the distributions file).

### 2.3 Data split

- Random image-level split, 80 / 10 / 10, fixed seed **42**.
- Expected sizes: training 8,058, validation 1,007, test 1,008 (confirm from code).
- Image-level split is safe: KonIQ-10k images are distinct and duplicates were removed when the dataset was built.
- Split files: `data/splits/train.csv`, `val.csv`, `test.csv` (columns: `image_name`, `mos`).
- MOS statistics of each split are compared with the full dataset after splitting.
- The test set is used **once**, for the Final Project Report only.

### 2.4 Preprocessing (identical in training and in the app)

1. Read image as RGB (convert from BGR if OpenCV is used).
2. Resize to **224 × 224** (configurable; 384 × 512 native resolution is an optional extension).
3. Cast to float and apply `tf.keras.applications.mobilenet_v2.preprocess_input` (scales to [−1, 1]).
4. Target: `mos_norm = MOS / 100` for training; predictions converted back with `MOS = 100 · mos_norm`. Displayed score out of 100 = `mos_norm × 100`, which on this scale is the MOS rounded. The code reads the scale from the score file rather than assuming it, so a 1-5 file would still work.
5. Augmentation: random horizontal flip on training images only. Nothing else.

All of this lives in one shared function in `src/preprocess.py`.

### 2.5 Model

| Stage | Operation | Output shape | Parameters |
|---|---|---|---|
| Input | 224 × 224 × 3 RGB | 224 × 224 × 3 | 0 |
| Backbone | MobileNetV2, ImageNet weights, `include_top=False` | 7 × 7 × 1280 | ≈ 2.26 M |
| Pooling | GlobalAveragePooling2D | 1280 | 0 |
| Dense | 128 units, ReLU | 128 | 163,968 |
| Dropout | rate 0.3 (starting value) | 128 | 0 |
| Output | Dense(1), linear | 1 | 129 |

- The backbone is called with `training=False` so batch-normalization layers stay in inference mode, including during fine-tuning.
- The Keras MobileNetV2 base has 154 layers.

### 2.6 Training

| Phase | Trainable | Learning rate (start) | Notes |
|---|---|---|---|
| 1 | Regression head only | 1e-3 | Backbone frozen |
| 2 | Head + backbone layers from index 100 onward | 1e-5 | Recompile after unfreezing |

- Optimizer: Adam. Loss: MSE. Tracked metric: MAE.
- Callbacks: ModelCheckpoint (best `val_loss`), EarlyStopping (restore best weights), CSVLogger to `outputs/logs/`.
- Batch size and epochs: set from actual runs (starting point: batch 32, up to 15 epochs per phase).
- Only values actually used may appear in Reports 3 and 4.

### 2.7 Evaluation

- MAE and MSE on the MOS scale (0–100).
- SROCC (`scipy.stats.spearmanr`) and PLCC (`scipy.stats.pearsonr`).
- Plots: predicted vs true scatter, loss curves, error by MOS range (low / mid / high).
- Plain Pearson is used for PLCC (no logistic mapping); state this in reports.

### 2.8 Quality levels and diagnostics

- Levels: **Poor / Average / Good**, thresholds = 33rd and 67th percentiles (tertiles) of the **training-set** MOS. Saved to `models/label_thresholds.json`.
- Diagnostics (OpenCV, on the image resized to 512 × 384, grayscale): sharpness = variance of Laplacian, brightness = mean intensity, contrast = standard deviation of intensity.
- Diagnostic bands (Low / Normal / High) from the 33rd and 67th percentiles over training images. Saved to `models/diagnostic_thresholds.json`.
- Diagnostics are shown as auxiliary information only; the model does not use them.

### 2.9 Saved artifacts

| File | Content |
|---|---|
| `models/iqa_mobilenetv2.keras` | Final trained model |
| `models/label_thresholds.json` | Poor/Average/Good MOS thresholds |
| `models/diagnostic_thresholds.json` | Sharpness/brightness/contrast bands |
| `outputs/logs/phase1_log.csv`, `phase2_log.csv` | Per-epoch training logs |
| `outputs/results/*.json` | Dataset stats, split stats, validation and test metrics |
| `outputs/figures/*.png` | All plots used in reports |

TensorFlow version in `requirements.txt` must match the Colab version used for training, or the `.keras` file may not load locally.

---

## 3. Folder structure

```
image-quality-assessment/
├── REPORTS_GUIDE.md                this file (the only markdown file)
├── requirements.txt                pinned libraries
├── app.py                          Streamlit demo (runs locally)
├── notebooks/
│   └── iqa_training_colab.ipynb    Colab: data, training, evaluation, export
│                                   (complete working code; src/ is refactored from it)
├── src/
│   ├── __init__.py
│   ├── config.py                   paths, seed, image size, hyperparameters
│   ├── data.py                     download, verify, load scores, split, statistics
│   ├── preprocess.py               shared image preprocessing + tf.data pipeline
│   ├── model.py                    MobileNetV2 regression model, unfreezing
│   ├── train.py                    two-phase training, callbacks, thresholds
│   ├── evaluate.py                 MAE, MSE, SROCC, PLCC, plots
│   ├── predict.py                  single-image inference for the app
│   └── quality_metrics.py          OpenCV diagnostics
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_metrics.py
├── data/
│   ├── raw/                        KonIQ-10k images + scores CSV (not committed)
│   └── splits/                     train.csv, val.csv, test.csv
├── models/                         trained model + threshold JSON files
├── outputs/
│   ├── figures/                    generated plots
│   ├── logs/                       training logs
│   └── results/                    metrics and statistics (JSON)
├── sample_images/                  demo photos (good / average / poor)
└── reports/
    ├── progress_report_1/          Report 1 (.docx, .pdf), figures/, evidence/
    ├── progress_report_2/          figures/, evidence/
    ├── progress_report_3/          figures/, evidence/
    └── final_report/               figures/, evidence/
```

**Notebook sections and the report that describes them**

| # | Notebook section | Report |
|---|---|---|
| 1 | Setup: clone repo, install, GPU and TF version check | 2 |
| 2 | Download and verify KonIQ-10k | 2 |
| 3 | MOS distribution analysis | 2 |
| 4 | Train / validation / test split | 2 |
| 5 | tf.data pipeline and visual check | 2 |
| 6 | Build model and summary | 2 |
| 7 | Phase 1 training | 3 |
| 8 | Phase 2 fine-tuning | 3 |
| 9 | Validation evaluation and plots | 3 |
| 10 | Label and diagnostic thresholds | 3 |
| 11 | Test evaluation (run once) | 4 |
| 12 | Export model and artifacts | 4 |

Not committed to GitHub: `data/raw/`, large model files if over 100 MB, Colab credentials.

---

## 4. Common format (all reports)

- A4, 1-inch margins; Times New Roman 12 pt, line spacing 1.15–1.25, justified.
- Headings numbered 1, 1.1 …; level 1 14 pt bold, level 2 12 pt bold.
- Header (right, 9 pt, grey): `Image Quality Assessment Using Deep Learning | <Report name>`
- Footer (centre, 9 pt): `Page X of Y`
- Title block for Reports 1–3: Report, Submitted to, Submitted by, Registration No., Date of submission. Report 4 uses a full title page.
- Table captions above tables (**Table N.**), figure captions below figures (**Figure N.**), numbered separately and referred to in the text before they appear. Tables must not break across pages.
- Figures as PNG, at least 200 dpi, readable labels. Diagrams drawn properly, not typed with arrow characters.
- Equations centred and numbered on the right; every symbol defined.
- References in IEEE style, numbered in order of first citation; every reference cited in the text.
- File names: `Progress_Report_1.docx/.pdf`, `Progress_Report_2…`, `Progress_Report_3…`, `Final_Project_Report…`

---

## 5. Writing style

**Do**
- Write in first person plural and give the reason for each decision.
- Use exact numbers from `outputs/` and the notebook.
- Describe problems honestly: what happened, the cause, the fix.
- Write mostly in prose; use numbered lists only for real steps.
- End every progress report with a Conclusion in prose.

**Do not**
- Mention weeks, timelines or plans by date.
- Include status tables, "completed / pending" labels, checklists or "next steps" bullet lists in a report.
- Report any number not produced by the code.
- Mention other, unrelated projects or earlier drafts.
- Use filler words: *delve, leverage, robust, seamless, comprehensive, crucial, pivotal, cutting-edge, utilize, plays a vital role, in today's world, it is worth noting.*
- Use long dashes (—) as connectors, spaced hyphens as dashes, emojis, or bold words inside paragraphs.
- Say "supplied paper"; call it "the reference paper for this project [n]".

---

## 6. Reference bank

Verified entries; renumber in order of citation.

- Q. Sang, Q. Li, L. Liu, Z. Deng, X. Wu, and A. C. Bovik, "No-reference image quality assessment leveraging GenAI images," *IEEE Trans. Image Process.*, vol. 34, pp. 6204–6214, 2025.
- V. Hosu, H. Lin, T. Sziranyi, and D. Saupe, "KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment," *IEEE Trans. Image Process.*, vol. 29, pp. 4041–4056, 2020.
- M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE/CVF CVPR*, 2018, pp. 4510–4520.
- J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE CVPR*, 2009, pp. 248–255.
- A. Mittal, A. K. Moorthy, and A. C. Bovik, "No-reference image quality assessment in the spatial domain," *IEEE Trans. Image Process.*, vol. 21, no. 12, pp. 4695–4708, 2012.
- A. Mittal, R. Soundararajan, and A. C. Bovik, "Making a 'completely blind' image quality analyzer," *IEEE Signal Process. Lett.*, vol. 20, no. 3, pp. 209–212, 2013.
- M. Abadi et al., "TensorFlow: Large-scale machine learning on heterogeneous systems," 2015. [Online]. Available: https://www.tensorflow.org
- D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015.

Check any new reference (authors, title, venue, year, pages) before adding it.

---

## 7. Report specifications

### Report 1: Progress Report 1 (Problem Study, Dataset and Model Design)

Already written: `reports/progress_report_1/`. Its sections: 1 Introduction, 2 Objectives, 3 Background Study (IQA and NR-IQA, deep learning for quality, MobileNetV2 and transfer learning, loss and metrics with equations), 4 Dataset and Data Preparation, 5 Model Design (feature extractor, regression head, training strategy, inference and quality levels), 6 Observations and Expected Outcomes, 7 Conclusion, References. Tables 1–6, Figures 1–3, Equations 1–4.

If any fact in Section 2 changes later, Reports 2–4 must use the new value and say briefly why it changed.

---

### Report 2: Progress Report 2 (Data Pipeline and Model Implementation)

**Purpose:** show that the data is verified and the pipeline and model are implemented and working together.
**Length:** 8–11 pages.

**Sections**
1. **Introduction**: one paragraph recalling Progress Report 1, then the scope of this report.
2. **Dataset Verification**: download source, file counts, CSV columns, missing or invalid rows; MOS statistics table (count, min, max, mean, std, median); MOS histogram; sample images from low, middle and high MOS with their scores. This report must also state that the score file publishes MOS on a 0-100 scale rather than the 1-5 scale given in Progress Report 1, and that every score, threshold and error figure from here on is on the 0-100 scale.
3. **Data Split**: actual split sizes table; MOS statistics per split table; overlaid histograms; why an image-level split is valid here.
4. **Preprocessing and Input Pipeline**: the shared preprocessing function; tf.data steps (decode, resize, preprocess, flip, batch, prefetch); MOS normalization; figure of preprocessed samples; how the same function is reused by the app.
5. **Model Implementation**: how the backbone and head are built; parameter table from `model.summary()` (total, trainable, non-trainable with frozen backbone); short code excerpt (≤ 15 lines) only if it explains something.
6. **Sanity Checks**: batch shapes table; untrained output range; loss on one batch; small overfitting test on a few dozen images showing the loss falling; metric function checks on known inputs.
7. **Problems Found and Fixed**: each real issue with cause and fix.
8. **Conclusion**
9. **References**

**Required tables:** MOS statistics, split sizes, per-split MOS statistics, parameter counts, sanity checks.
**Required figures:** MOS histogram, sample images by quality, preprocessed batch, split histograms.

**Evidence checklist**
- [ ] `outputs/results/dataset_stats.json` and printed file counts
- [ ] `outputs/results/split_stats.json` and split CSV row counts
- [ ] MOS histogram and sample-image figures
- [ ] Printed `model.summary()` and parameter counts
- [ ] Batch shape printout and one-batch loss
- [ ] Small overfitting test output
- [ ] TensorFlow version and GPU name from Colab
- [ ] Notes on every error met

---

### Report 3: Progress Report 3 (Model Training and Validation)

**Purpose:** show how the model was trained, how it performs on validation data, and how the presentation thresholds were set.
**Length:** 8–12 pages.

**Sections**
1. **Introduction**: one paragraph recalling Progress Report 2, then the scope.
2. **Training Configuration**: table of actual values (GPU, TF version, image size, batch size, epochs per phase, learning rates, unfreeze index, dropout, callbacks, seed); training time.
3. **Phase 1: Training the Regression Head**: loss and MAE curves; best epoch; validation MAE, SROCC, PLCC.
4. **Phase 2: Fine-Tuning**: what was unfrozen and why; curves; best epoch; comparison table of validation metrics after Phase 1 and after Phase 2.
5. **Validation Analysis**: predicted vs true scatter; MAE by MOS range (low / mid / high) table; examples of good and poor predictions with images.
6. **Quality Levels and Diagnostics**: tertile thresholds table; diagnostic band table; how they were computed from training data only.
7. **Problems Found and Fixed**: training errors, instability, overfitting signs and what was done.
8. **Conclusion**
9. **References**

**Required tables:** training configuration, Phase 1 vs Phase 2 validation metrics, MAE by MOS range, thresholds.
**Required figures:** Phase 1 curves, Phase 2 curves, validation scatter plot, example predictions.

**Evidence checklist**
- [ ] `outputs/logs/phase1_log.csv`, `phase2_log.csv`
- [ ] Curve figures from `outputs/figures/`
- [ ] `outputs/results/val_metrics.json` (after each phase)
- [ ] Validation scatter plot and MAE-by-range output
- [ ] `models/label_thresholds.json`, `models/diagnostic_thresholds.json`
- [ ] Training time and GPU name
- [ ] Full text of any error and its fix

---

### Report 4: Final Project Report

**Purpose:** the complete, self-contained account of the project.
**Length:** 30–45 pages (excluding appendices).

**Front matter:** title page (institution, department, title, "Final Project Report", student, registration number, supervisor, month and year); certificate and declaration if the department requires them; acknowledgement; abstract (200–250 words with the final test SROCC, PLCC and MAE); table of contents; lists of figures, tables and abbreviations.

**Chapters**
1. **Introduction**: background, problem statement, motivation, objectives, scope and limitations, organization.
2. **Literature Review**: FR, RR and NR-IQA; classical NR-IQA (BRISQUE, NIQE); CNN-based and transfer-learning approaches; IQA datasets (synthetic vs authentic); the reference paper and why this project uses a lighter design; comparison table.
3. **Dataset**: KonIQ-10k description, verification, MOS distribution, split, preprocessing.
4. **Methodology**: pipeline figure, architecture, loss, metrics with equations, two-phase training, quality levels and diagnostics.
5. **Implementation**: environment, code organization, data pipeline, model, training, inference module, Streamlit application with screenshots, testing, problems and fixes.
6. **Results and Discussion**:
   - Training behaviour (curves).
   - **Test-set results table**: MAE, MSE, SROCC, PLCC, with validation results alongside.
   - Predicted vs true scatter on the test set; error by MOS range.
   - Demo: at least three real photographs (good, average, poor) with score, level and diagnostics, shown as app screenshots.
   - Context: published KonIQ-10k results (e.g. KonCept512) use larger backbones and higher resolution, so they are not directly comparable; say so if cited.
   - Limitations: 224 × 224 resizing, few extreme-MOS images, authentic-only training data.
7. **Conclusion and Future Work**: achievements against each objective; extensions (native-resolution training, other backbones, more datasets, deployment).

**References**, then **Appendices**: A environment and hyperparameters, B extra predictions, C app usage.

**Rules:** rewrite, do not paste, material from Reports 1–3; test numbers appear only here; all numbers match the final run.

**Evidence checklist**
- [ ] Everything from Reports 2 and 3 (updated if changed)
- [ ] `outputs/results/test_metrics.json`
- [ ] Test scatter plot and error-by-range output
- [ ] App screenshots for the three demo images
- [ ] Final `requirements.txt`
- [ ] Department title page / certificate format, if any

---

## 8. Final checks before submitting

- [ ] No weeks, timelines, status tables or next-step lists
- [ ] No mention of unrelated projects
- [ ] Every number traced to `outputs/`, `models/` or a notebook printout
- [ ] Facts match Section 2
- [ ] Tables and figures numbered, captioned, referred to
- [ ] All equation symbols defined
- [ ] Every citation in the list and every listed reference cited
- [ ] Title block complete
- [ ] Saved as `.docx` and `.pdf` with the right name

---

## 9. Prompt for generating a report

```
Read REPORTS_GUIDE.md fully. Write [Progress Report 2 / Progress Report 3 /
the Final Project Report] exactly as specified in Section 7, following the
format in Section 4 and the style in Section 5, and consistent with the
project facts in Section 2. Match the look of
reports/progress_report_1/Progress_Report_1.pdf.

Use only the evidence below for numbers and results. If something the
specification needs is missing, list it at the end instead of inventing it.

Evidence:
[paste printed outputs, JSON contents, logs, errors and notes]

Figures attached:
[file name – what it shows]

Student details: Name [ ]  Registration No. [ ]  Date of submission [ ]

Output: A4 Word document (.docx) and a PDF copy.
```
