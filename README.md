# D33 — SVM & KNN Assignments
**IIT Gandhinagar | PG Diploma in AI-ML & Agentic AI Engineering | Week 6**

---

## Repository Structuresss

```
SVM_KNN_Assignment/
├── AM/
│   ├── PartA_AM_SVM_KNN_DigitClassifier.ipynb
│   ├── PartB_AM_FAISS_ANN.ipynb
│   ├── PartC_AM_Interview_Questions.ipynb
│   └── PartD_AM_Visualization_KernelTrick.ipynb
├── PM/
│   ├── PartA_PM_8Algorithm_CheatSheet.ipynb
│   ├── PartB_PM_TextClassification_TfIdf_SVM.ipynb
│   ├── PartC_PM_Interview_Questions.ipynb
│   └── PartD_PM_AlgorithmSelectionGuide.ipynb
└── README.md
```

---

## AM Session — SVM & KNN on MNIST Digits

**Theme:** Build a handwritten digit classifier using SVM and KNN on sklearn's `load_digits` dataset.

### Part A — Concept Application (40%)
**File:** `PartA_AM_SVM_KNN_DigitClassifier.ipynb`

Covers the full ML pipeline for digit classification:

1. **Load & Explore** — `sklearn.datasets.load_digits` (1797 samples, 64 features, 10 classes). Visualises two sample images per digit.
2. **Feature Scaling** — `StandardScaler` fitted on training set only; applied to both train and test (prevents data leakage).
3. **SVM (RBF) + GridSearchCV** — Grid over `C ∈ {0.1, 1, 10, 100}` and `gamma ∈ {0.001, 0.01, 0.1, 1}` with 5-fold CV. Includes heatmap of CV scores.
4. **KNN — Optimal K** — Plots CV accuracy for K = 1…20; selects the elbow point.
5. **Comparison** — Side-by-side confusion matrices and per-class F1 bar chart (SVM vs KNN).
6. **Confusion Analysis** — `top_confused_pairs()` utility reveals the most common mis-classifications (e.g., 3→8, 4→9, 1→7).

**Expected Results:**
| Model | Accuracy |
|-------|----------|
| SVM (RBF, C=10, gamma=0.001) | ~0.98 |
| KNN (K=3) | ~0.97 |

---

### Part B — Stretch: FAISS Approximate Nearest Neighbours (30%)
**File:** `PartB_AM_FAISS_ANN.ipynb`

Self-study on FAISS (Facebook AI Similarity Search):

- Installs `faiss-cpu` and builds a `IndexFlatL2` (exact) index on the digits training set.
- Implements majority-vote KNN classification via FAISS index search.
- **Speed benchmark** (1 000 queries): sklearn KNN vs FAISS — timed with `time.perf_counter()`.
- **Findings documented:** FAISS achieves same accuracy as sklearn with significant speedup due to optimised C++/SIMD code. Approximate indexes (`IndexIVFFlat`, `IndexHNSW`) push this further.

**Why FAISS matters:** Powers similarity search at Instagram (1B+ photos), Spotify recommendations, and LLM RAG retrieval systems.

---

### Part C — Interview Ready (20%)
**File:** `PartC_AM_Interview_Questions.ipynb`

Three interview-style questions answered in full:

**Q1 — Conceptual: SVM vs Logistic Regression**
- LR minimises log-loss over all points; SVM maximises the margin using only support vectors.
- LR gives calibrated probabilities; SVM uses hinge loss.
- Decision table: when to use LR (need probabilities, large data) vs SVM (high-dim, kernel tricks needed).

**Q2 — Coding: `knn_from_scratch(X_train, y_train, X_test, k)`**
- Pure NumPy implementation using vectorised Euclidean distance.
- Majority vote using `np.bincount().argmax()`.
- Validated against sklearn KNN — predictions match exactly.

**Q3 — Debug: SVM scoring 0.50**
```python
svm = SVC(kernel='rbf', C=1.0)
svm.fit(X_train, y_train)  # salary (50K-200K), age (20-60)
# → 0.50 accuracy
```
**Root cause:** Missing `StandardScaler`. Salary (50K range) dominates the RBF kernel's distance computation; `age` becomes invisible. Fix: wrap in `Pipeline([('scaler', StandardScaler()), ('svm', SVC(...))])`.

---

### Part D — AI-Augmented: Visualization & Kernel Trick (10%)
**File:** `PartD_AM_Visualization_KernelTrick.ipynb`

**Visualization (Steps 12-13):**
- `make_moons` dataset with 5 values of C: `{0.01, 0.1, 1, 10, 100}`.
- Each subplot shows decision boundary, ±1 margin (dashed), and circled support vectors.
- Dual plot: # support vectors vs train accuracy as C increases.
- **Insight:** Small C → wide margin, more SVs, softer boundary. Large C → narrow margin, fewer SVs, overfit.

**Kernel Trick Analogy (Steps 14-15):**
> *"The Crumpled Paper"* — Two colours of marbles on a flat table can't be separated by a straight line. Lifting the tablecloth into 3D separates them — a flat card can now divide them. The kernel trick does this mathematically without ever explicitly constructing the 3D coordinates.

Analogy verified against mathematical definitions in a table (5/5 aspects accurate ✅).

---

## PM Session — Week 6 Full Comparison & SVM for Text

**Theme:** Build a comprehensive ML cheat sheet notebook and explore SVM for text classification.

### Part A — 8-Algorithm Cheat Sheet (40%)
**File:** `PartA_PM_8Algorithm_CheatSheet.ipynb`

**Algorithm Cards** (one per algorithm with: description, params, pros, cons, 5-line code):

| # | Algorithm | Best For |
|---|-----------|----------|
| 1 | Logistic Regression | Baseline, probability output, large data |
| 2 | Decision Tree | Interpretability, mixed types, no scaling |
| 3 | Random Forest | General-purpose, feature importance |
| 4 | Gradient Boosting | Best accuracy on tabular data |
| 5 | SVM (RBF) | High-dim, small-medium datasets |
| 6 | KNN | Instance-based, similarity tasks |
| 7 | Naive Bayes | Text, very small datasets |
| 8 | XGBoost | Kaggle-style tabular tasks |

**Head-to-head comparison** on `load_breast_cancer` (30 features, 569 samples):
- All 8 models run with identical 5-fold `StratifiedKFold` and `StandardScaler`.
- Results ranked by mean CV accuracy with error bars.
- **Recommendation:** GBM/XGBoost tops accuracy; Logistic Regression recommended for medical settings requiring interpretability.

---

### Part B — Stretch: TF-IDF + SVM Text Classification (30%)
**File:** `PartB_PM_TextClassification_TfIdf_SVM.ipynb`

Classic text classification pipeline:

- **Dataset:** `fetch_20newsgroups` — 4 categories: `sci.space`, `rec.sport.hockey`, `talk.politics.guns`, `comp.graphics`.
- **SVM Pipeline:** `TfidfVectorizer(max_features=20K, ngram_range=(1,2), sublinear_tf=True)` → `LinearSVC(C=1.0)`.
- **LR Pipeline:** Same TF-IDF → `LogisticRegression(C=5.0, solver='saga')`.
- Outputs: accuracy, full `classification_report`, and side-by-side confusion matrices.

**Key finding:** LinearSVC and Logistic Regression both achieve ~95%+ accuracy. LinearSVC is faster at inference; LR gives probability scores.

**Why it matters:** SVM with linear kernel is the classic text classification algorithm used at Google (spam), Yahoo News, and Bloomberg (news categorisation).

---

### Part C — Interview Ready (20%)
**File:** `PartC_PM_Interview_Questions.ipynb`

**Q1 — Conceptual: 100 features, 50 samples**

Full analysis table of all 8 algorithms in the p>>n regime:
- ✅ Works: Logistic Regression (L1), LinearSVC, Naive Bayes
- ⚠️ Risky: SVM (RBF), Random Forest
- ❌ Fails: Decision Tree, KNN, Gradient Boosting / XGBoost

**Q2 — Coding: `model_selection_report(X, y, models_dict)`**
- Splits data, applies StandardScaler inside each fold (no leakage).
- Runs 5-fold CV for every model using `cross_val_score`.
- Returns a ranked `pd.DataFrame` with: Mean, Std, Min, Max, 95% CI.
- **Statistical test:** Paired t-test (scipy) of each model against the best; flags significance at α=0.05.

**Q3 — Analyze: Train=1.0, Test=0.52**

Three specific fixes:
1. Reduce `C` (soften margin — `C=100 → C=1`)
2. Reduce `gamma` (wider kernel — `gamma=10 → gamma='scale'`)
3. `GridSearchCV` on both `C` and `gamma` jointly with 5-fold CV

---

### Part D — AI-Augmented: Algorithm Selection Guide (10%)
**File:** `PartD_PM_AlgorithmSelectionGuide.ipynb`

**`recommend_algorithm()` function** — rule-based decision engine:
- Inputs: `n_samples`, `n_features`, `is_linear`, `need_probability`, `need_interpretability`, `is_text`
- Outputs: ranked recommendations + models to avoid + reasoning notes

**Tested on 4 scenarios:**
1. Medical diagnosis → Logistic Regression (interpretability + probabilities)
2. Digit recognition → SVM (RBF) / Random Forest
3. Text spam detection → LinearSVC + TF-IDF
4. Gene expression (p>>n) → Logistic Regression (L1)

**Edge cases the AI missed (verified against experience):**
- Imbalanced classes → `class_weight='balanced'` + AUC metric
- Mixed feature types → tree models preferred over SVM
- Multi-label problems → `OneVsRestClassifier`
- Streaming data → `SGDClassifier`
- Explainability for black-box models → SHAP values

**Personal reference table** saved at the end of the notebook.

---

## Setup & Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install faiss-cpu          # Part B AM
pip install xgboost            # Part A PM (optional, falls back to GBM)
pip install scipy              # Part C PM (paired t-test)
```

**Python:** 3.8+  
**Jupyter:** Classic Notebook or JupyterLab

---

## Quick Start

```bash
# Clone your repo and navigate in
cd your-repo/

# Start Jupyter
jupyter notebook

# Open notebooks in order:
# AM: PartA → PartB → PartC → PartD
# PM: PartA → PartB → PartC → PartD
```

---

## Key Concepts Covered

| Topic | Where |
|-------|-------|
| SVM hard/soft margin, C parameter | AM Part A, D |
| RBF kernel, gamma parameter | AM Part A, D |
| Support vectors | AM Part D |
| KNN, distance metrics, choosing K | AM Part A |
| Curse of dimensionality | PM Part C Q1 |
| Scaling importance | AM Part C Q3, PM Part C Q3 |
| FAISS / Approximate NN | AM Part B |
| TF-IDF + LinearSVC pipeline | PM Part B |
| Overfitting diagnosis & fixes | PM Part C Q3 |
| Algorithm selection framework | PM Part D |
| Kernel trick | AM Part D |
| SVM vs Logistic Regression | AM Part C Q1 |

---

*IIT Gandhinagar | PG Diploma in AI-ML & Agentic AI Engineering | Confidential*
