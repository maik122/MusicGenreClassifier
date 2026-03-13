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
| 🔴 Must | Proper train / val / test three-way split | ✅ Done |
| 🟠 Should | Config file for all hardcoded paths and params | ✅ Done |
| 🟠 Should | Validate image directory structure for ResNet (`class/image.jpg`) | ✅ Done |
| 🟡 Could | Augment spectrogram images (flip, brightness, zoom) | ⏳ Later |
| 🟡 Could | Extract custom MFCC / chroma features with `librosa` | ✅ Done |

---

### Modelling & Evaluation

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | Random Forest classifier (tabular features) | ✅ Done |
| 🔴 Must | Fix CNN — fine-tune ResNet50 last 20 layers at `lr=1e-5` | ⏳ In Progress |
| 🔴 Must | Confusion matrix + classification report | ✅ Done |
| 🟠 Should | K-fold cross-validation on Random Forest | ⏳ Todo |
| 🟠 Should | Hyperparameter tuning with `Optuna` or `GridSearchCV` | ⏳ Todo |
| 🟠 Should | XGBoost / LightGBM as stronger tabular baseline | ⏳ Todo |
| 🟠 Should | Persist models to disk (`joblib`, `model.save()`) | ✅ Done |
| 🟡 Could | Feature importance plot (which audio features matter most) | ⏳ Later |
| 🟡 Could | Probability calibration on classifier outputs | ⏳ Later |

---

### Demo & Deployment

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | Inference pipeline — load saved model and predict on new audio | ✅ Done |
| 🟠 Should | Streamlit app — upload audio → genre prediction | ✅ Done |
| 🟠 Should | Deploy demo to Hugging Face Spaces (free, shareable link) | ⏳ Next |
| 🟡 Could | Docker containerisation | ⏳ Later |
| 🟡 Could | FastAPI `/predict` endpoint | ⏳ Later |
| ⚪ Won't | Real-time microphone inference | — |
| ⚪ Won't | Mobile or desktop app packaging | — |

---

### Code Quality & Structure

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | Refactor flat script into modules (`src/config`, `src/predict`) | ✅ Done |
| 🔴 Must | Replace all `print` statements with `logging` | ✅ Done |
| 🔴 Must | Pin dependencies in `requirements.txt` | ✅ Done |
| 🟠 Should | Docstrings on every function and class | ✅ Done |
| 🟠 Should | `config.py` for all magic numbers and paths | ✅ Done |
| 🟡 Could | Unit tests for feature extraction and preprocessing (`tests/`) | ⏳ Later |
| ⚪ Won't | CI/CD pipeline | — |

---

### Documentation & Portfolio

| Priority | Task | Status |
|----------|------|--------|
| 🔴 Must | README — problem statement, dataset, how to run | ⏳ Next |
| 🔴 Must | Results table comparing Random Forest vs CNN | ⏳ Next |
| 🟠 Should | Architecture diagram (tabular pipeline vs spectrogram pipeline) | ⏳ Todo |
| 🟠 Should | Analytical narrative — why classical is easy, why rock/blues overlap | ⏳ Todo |
| 🟡 Could | Grad-CAM saliency maps on misclassified spectrograms | ⏳ Later |
| ⚪ Won't | Multi-label classification (songs spanning genres) | — |
| ⚪ Won't | Custom dataset collection beyond GTZAN | — |

---

## Next Steps (Priority Order)

2. **Write the README** — problem statement, results table, how to run, live demo link
3. **Fix the CNN properly** — currently removed from app; resolve TensorFlow/Keras version issues and get it predicting
4. **Cross-validation + XGBoost** — strengthens the ML narrative with rigour
5. **Architecture diagram** — visual showing tabular vs spectrogram pipeline side by side