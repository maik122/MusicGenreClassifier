# Music Genre Classifier — Project Roadmap

## MoSCoW Prioritisation

**Priority Key:** 🔴 Must Have | 🟠 Should Have | 🟡 Could Have | ⚪ Won't Have (now)

---

### Data & Features

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | Load GTZAN CSV features (`features_30_sec.csv`) | ✅ Done |
| 🔴 Must | Encode genre labels with `LabelEncoder` | ✅ Done |
| 🔴 Must | Scale features with `StandardScaler` | ✅ Done |
| 🔴 Must | Proper train / val / test three-way split | ⏳ Todo |
| 🟠 Should | Config file for all hardcoded paths and params | ⏳ Todo |
| 🟠 Should | Validate image directory structure for ResNet (`class/image.jpg`) | ⏳ Todo |
| 🟡 Could | Augment spectrogram images (flip, brightness, zoom) | ⏳ Later |
| 🟡 Could | Extract custom MFCC / chroma features with `librosa` | ⏳ Later |

---

### Modelling & Evaluation

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | Random Forest classifier (tabular features) | ✅ Done |
| 🔴 Must | Fix CNN — fine-tune ResNet50 last 20 layers at `lr=1e-5` | ⏳ Next |
| 🔴 Must | Confusion matrix + classification report | ✅ Done |
| 🟠 Should | K-fold cross-validation on Random Forest | ⏳ Todo |
| 🟠 Should | Hyperparameter tuning with `Optuna` or `GridSearchCV` | ⏳ Todo |
| 🟠 Should | XGBoost / LightGBM as stronger tabular baseline | ⏳ Todo |
| 🟠 Should | Persist models to disk (`joblib`, `model.save()`) | ⏳ Todo |
| 🟡 Could | Feature importance plot (which audio features matter most) | ⏳ Later |
| 🟡 Could | Probability calibration on classifier outputs | ⏳ Later |

---

### Demo & Deployment

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | Inference pipeline — load saved model and predict on new audio | ⏳ Next |
| 🟠 Should | Gradio or Streamlit app — upload audio → genre prediction | ⏳ Todo |
| 🟠 Should | Deploy demo to Hugging Face Spaces (free, shareable link) | ⏳ Todo |
| 🟡 Could | Docker containerisation | ⏳ Later |
| 🟡 Could | FastAPI `/predict` endpoint | ⏳ Later |
| ⚪ Won't | Real-time microphone inference | — |
| ⚪ Won't | Mobile or desktop app packaging | — |

---

### Code Quality & Structure

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | Refactor flat script into modules (`src/data`, `src/models`, `src/evaluate`, `src/predict`) | ⏳ Next |
| 🔴 Must | Replace all `print` statements with `logging` | ⏳ Todo |
| 🔴 Must | Pin dependencies in `requirements.txt` | ⏳ Todo |
| 🟠 Should | Docstrings on every function and class | ⏳ Todo |
| 🟠 Should | `config.yaml` for all magic numbers and paths | ⏳ Todo |
| 🟡 Could | Unit tests for feature extraction and preprocessing (`tests/`) | ⏳ Later |
| ⚪ Won't | CI/CD pipeline | — |

---

### Documentation & Portfolio

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | README — problem statement, dataset, how to run | ⏳ Todo |
| 🔴 Must | Results table comparing Random Forest vs CNN | ⏳ Todo |
| 🟠 Should | Architecture diagram (tabular pipeline vs spectrogram pipeline) | ⏳ Todo |
| 🟠 Should | Analytical narrative — why classical is easy, why rock/blues overlap | ⏳ Todo |
| 🟡 Could | Grad-CAM saliency maps on misclassified spectrograms | ⏳ Later |
| ⚪ Won't | Multi-label classification (songs spanning genres) | — |
| ⚪ Won't | Custom dataset collection beyond GTZAN | — |

---

## Next Steps (Priority Order)

1. **Fix the CNN** — fine-tune ResNet50 last 20 layers; 0.20 accuracy is a red flag
2. **Build inference pipeline** — load saved models and predict on a new audio file end-to-end
3. **Refactor into modules** — shows software engineering maturity to reviewers
4. **Persist models to disk** — project must be runnable without retraining
5. **Gradio demo + Hugging Face deploy** — highest-impact portfolio item; tangible and shareable
6. **Cross-validation + XGBoost** — strengthens the ML narrative with rigour