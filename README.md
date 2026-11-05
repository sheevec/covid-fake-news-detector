# COVID-19 Fake News Detector

![Python](https://img.shields.io/badge/Python-3.8-blue) ![AWS](https://img.shields.io/badge/AWS-SageMaker-FF9900) ![scikit--learn](https://img.shields.io/badge/scikit--learn-0.24-F7931E) ![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20TextBlob-green)

> **NLP-powered misinformation classifier** built as a team project (Husky 4) at Northeastern University. Detects COVID-19 fake news using linguistic feature engineering + AWS SageMaker Linear Learner and scikit-learn SVM pipelines.

---

## Overview

During the COVID-19 pandemic, misinformation spread rapidly across social media and news outlets. This project builds a binary classifier to automatically detect fake news articles using a rich set of linguistic and stylistic features — without relying on pre-trained language models.

The pipeline is designed to run as an **AWS SageMaker training job**, with a custom scikit-learn preprocessor and a Linear Learner estimator for scalable inference.

---

## Architecture

```
Raw Articles (title, text, source)
        │
        ▼
┌─────────────────────────────────────┐
│       Feature Engineering           │
│  • Uppercase ratio (title/text)     │
│  • Stop word density                │
│  • POS tag distribution             │
│  • Negation & interrogative counts  │
│  • Readability scores (Flesch, SMOG,│
│    Gunning Fog, Coleman-Liau, etc.) │
│  • Lexical richness (TTR)           │
│  • Power / causal / emotion words   │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│   scikit-learn Preprocessing        │
│   StandardScaler + ColumnTransformer│
└──────────────────┬──────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  LinearSVC (local)    AWS SageMaker
  cross-validation     Linear Learner
```

---

## Dataset

- **Source:** `corona_fake.csv` — COVID-19 articles labeled FAKE / TRUE
- **Size:** 1,164 rows; 1,159 labeled (575 fake, 584 true) after normalising label spellings (`Fake` / `fake` → `FAKE`)
- **Features used:** `title`, `text`, `source`
- **Label:** `label` (binary: FAKE / TRUE)
- Articles from 269 sources, including Facebook, news sites, health agencies and social media

---

## Key Results

Re-run on all 1,159 labeled articles with the project's own feature code (63 features), 5-fold stratified cross-validation, fake as the positive class:

| Model | Accuracy | Precision | Recall | F1 | Notes |
|-------|----------|-----------|--------|----|-------|
| LinearSVC (linguistic features) | 83.8% | 83.2% | 84.4% | 0.838 | Project's scikit-learn model |
| Logistic regression (linguistic features) | 84.0% | 83.5% | 84.5% | 0.840 | Local stand-in for SageMaker Linear Learner |
| TF-IDF words + LinearSVC | 93.4% | 93.2% | 93.4% | 0.933 | Word-based comparison |

The SageMaker Linear Learner job needs an AWS account to re-run, so it is not included above.

### What separates fake from true articles

| Signal (average per article) | Fake vs true |
|---|---|
| Uppercase letters in the title | **+240%** (15.5 vs 4.6) |
| Uppercase share of the text | +52% |
| Article length | +34% |
| Power words per 100 words | −32% |
| Tentative words ("may", "could") per 100 words | −30% |
| Question words in the title | **−84%** |

### Caveat: labels follow the publisher

266 of the 269 sources publish only fake or only true articles. A model can therefore score well by recognising each publisher's house style rather than misinformation itself. A stricter evaluation would hold out entire sources (grouped cross-validation), so the model is tested on publishers it has never seen.

📊 **Interactive dashboard:** [shivamchaudhary.dev/dashboards/covid-fake-news](https://shivamchaudhary.dev/dashboards/covid-fake-news/)

---

## Feature Engineering Highlights

| Feature Category | Examples |
|-----------------|---------|
| **Stylistic** | `title_pct_uppercase`, `text_pct_uppercase` |
| **Lexical** | `ttr` (type-token ratio), `difficult_words` |
| **Readability** | Flesch Reading Ease, SMOG Index, Gunning Fog, Coleman-Liau |
| **Linguistic** | POS tag counts (NN, VB, JJ...), `num_negation`, `num_interrogatives` |
| **Sentiment** | `num_powerWords_text`, `num_emotionWords_text`, `num_tentativeWords_text` |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Feature Engineering | Python, NLTK, TextBlob, textstat, lexicalrichness |
| ML Pipeline | scikit-learn (ColumnTransformer, StandardScaler, LinearSVC) |
| Cloud Training | AWS SageMaker (custom script mode) |
| Inference | SageMaker Linear Learner endpoint |
| Data | pandas, numpy |

---

## Project Structure

```
covid-fake-news-detector/
├── src/
│   ├── sklearn_preprocessor.py          # SageMaker training script + feature engineering
│   ├── Inference Pipeline with          # Jupyter notebook: end-to-end inference pipeline
│   │   Scikit-learn and Linear Learner.ipynb
│   └── corona_fake.csv                  # COVID-19 fake news dataset
└── README.md
```

---

## Setup

```bash
pip install scikit-learn nltk textstat lexicalrichness textblob pandas numpy

# NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

# Run locally
python src/sklearn_preprocessor.py \
  --output-data-dir ./output \
  --model-dir ./model \
  --train ./src
```

For SageMaker deployment, upload the script to S3 and configure a `SKLearn` estimator pointing to `sklearn_preprocessor.py`.

---

## Academic Context

Built as a team project ("Husky 4" — Sagar, Shivam, Saju) for **EAI6010: Applications of AI** at Northeastern University. The focus was on classical NLP feature engineering rather than transformer-based approaches, demonstrating that interpretable linguistic features can achieve strong misinformation detection without black-box models.
