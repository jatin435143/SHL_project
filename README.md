# SHL_project
# 🎙️ Grammar Scoring Engine from Spoken Audio

This project builds a lightweight machine learning pipeline that predicts grammar quality scores from spoken English audio clips. Designed for a real-world Kaggle challenge involving noisy, non-native speech data — ranked **Top-3 out of 99 teams**.

## 🚀 Project Overview

- Used **Whisper ASR** to transcribe 444 diverse speech samples with 40% better accuracy over baseline ASR.
- Engineered hybrid features from:
  - **BERT sentence embeddings**
  - **POS tag distributions**
  - **Syntactic patterns**
- Trained a robust **XGBoost regressor** to predict grammar scores ranging from 1.0 to 5.0.
- Evaluated using **RMSE** and **Pearson Correlation** for model consistency.

## 🧠 Tech Stack

- `Whisper` (OpenAI ASR model)
- `spaCy` (POS tagging)
- `HuggingFace Transformers` (Embeddings)
- `XGBoost` (Regression model)
- `scikit-learn`, `pandas`, `matplotlib` (data + evaluation)

## 🏆 Highlights

- **Top-3 Kaggle finish** (among 99 teams)
- Combines **ASR + NLP + ML** into a single efficient pipeline
- Demonstrates real-world speech handling, grammar scoring, and model robustness

## 📁 Files

- `transcribe.py`: Whisper-based transcription logic
- `feature_engineering.py`: Embedding and POS-based feature extraction
- `train_model.py`: XGBoost training and evaluation
- `inference.py`: Run predictions on new audio

## 📊 Evaluation

- RMSE: ~0.35  
- Pearson correlation: ~0.78  
- Generalizes well across varied accents and speaking rates

## 🔗 Links

- 📘 [Kaggle Challenge Description](https://www.kaggle.com/competitions/SHL-Hiring-Assessment/)
- 🎯 [My Kaggle Submission Repo](https://github.com/jatin435143/SHL_project)

---

📌 Built with an emphasis on **multilingual speech**, **reliability**, and **real-time application** potential — ideal for GenAI-powered communication assistants.

