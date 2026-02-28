# NLP Assignment 2: Neural Sequence Models for Evasion Detection
> Neural Sequence Models for Automated Detection of Evasion Strategies in Political Discourse

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg) ![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg) ![License: Academic](https://img.shields.io/badge/license-Academic-green.svg)

**Authors:** Maria Goicoechea, Joaquin Orradre, Paula Pina
**Course:** Natural Language Processing | February 2026
**Paper:** [ACL Anthology — EMNLP 2024 Findings](https://aclanthology.org/2024.findings-emnlp.300/)

---

## 📖 Overview

This project extends the Assignment 1 baseline to neural sequence models for detecting evasion strategies in political interviews. We evaluate recurrent architectures (GRU, LSTM) trained from scratch and fine-tuned Transformer models (BERT, RoBERTa, DistilBERT) on the 9-class **QEvasion** dataset, comparing performance, learning curves, ablations, and computational cost against the classical CountVectorizer + Logistic Regression baseline (F1-Macro = 0.2823).

---

## 🎯 Task

**Evasion Classification (9 classes):** *Claims ignorance, Clarification, Declining to answer, Deflection, Dodging, Explicit, General, Implicit, Partial/half-answer*

Labels are consolidated by majority vote (ties broken randomly, seed 42) over three annotators. The dataset is heavily imbalanced: *Explicit* covers ~30% of training examples while *Partial/half-answer* has only 63.

---

## 📊 Dataset

- **Source:** [QEvasion](https://huggingface.co/datasets/ailsntua/QEvasion) (`ailsntua/QEvasion`)
- **Paper:** *"I Never Said That"* — Thomas et al., EMNLP 2024
- **Split used:** 2,758 train / 690 validation (80/20 from official train) / 308 held-out test
- **Input feature:** `question [SEP] interview_answer`

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/joaquinorradre/NLP_Assignment_2.git
cd NLP_Assignment_2
```

### Run Experiments
```bash
jupyter notebook notebooks/
```

---

## 🔬 Experiments

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | **Architecture Comparison** | Baseline vs. UniGRU vs. UniLSTM vs. BERT on full training set |
| 2 | **Learning Curve Analysis** | All models trained at 25%, 50%, 75%, 100% of data |
| 3 | **Ablation & Hyperparameter Tuning** | Frozen vs. unfrozen encoder; BERT vs. RoBERTa vs. DistilBERT; RoBERTa grid search |
| 4 | **Error Analysis** | Per-class metrics + confusion matrix + representative failure cases on test set |
| 5 | **Computational Cost Analysis** | Training time, inference speed, GPU memory, and parameter counts |

---


## 📝 Main Findings

1. **Fine-tuned Transformers outperform all other approaches.** RoBERTa achieves F1 = 0.3725, a +9.0 pp gain over the Assignment 1 baseline, driven by rich pre-trained contextual representations.
2. **Recurrent models trained from scratch do not justify added complexity** at this dataset size — both UniGRU and UniLSTM fall below the classical baseline. An estimated >10K examples would be needed to close the gap.
3. **Full fine-tuning is essential for Transformers.** Freezing the BERT encoder collapses F1 to 0.052 (random ≈ 0.11), confirming domain adaptation of all layers is required.
4. **DistilBERT offers the best efficiency/accuracy trade-off:** F1 = 0.3349 with 61% of BERT's parameters, 47% of its training time, and 2× faster inference.
5. **The task remains inherently difficult.** 67% of test examples are misclassified, mostly due to genuine linguistic ambiguity between categories (*Implicit/Explicit*, *Deflection/Dodging*).

---


## 🛠️ Technologies

- Python 3.8+
- PyTorch (recurrent models and training loops)
- HuggingFace Transformers and Datasets
- scikit-learn (baseline and evaluation)
- pandas and numpy (data processing)
- matplotlib and seaborn (visualization)
