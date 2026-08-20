# Emotion and Sentiment Classification on GoEmotions

**Author:** MD Rakib Hossain | **Student ID:** A00057300
**Module:** Data Analytics, University of Roehampton
**Dataset:** [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (Reddit comments labelled with 27 emotions plus neutral)

A text-classification project that turns raw Reddit comments into emotion and sentiment predictions using a carefully engineered NLP preprocessing pipeline with TF-IDF features and Logistic Regression. It compares two framings of the same data: fine-grained emotion classification and coarse 3-class sentiment.

## What This Project Does

The GoEmotions dataset is multi-label and noisy. This project first reduces it to a clean single-label problem (dropping rows with no emotion and rows annotators marked as very unclear, then taking the strongest emotion per comment), then builds a domain-aware text-cleaning pipeline and trains two classifiers:

1. **Emotion model:** predicts the fine-grained emotion label (the notebook's 29-class model, covering the GoEmotions emotion categories).
2. **Sentiment model:** groups those emotions into positive, negative, and neutral, then predicts the sentiment class.

## Preprocessing Pipeline

The text cleaning is the core of the project and is designed to preserve emotional signal rather than strip it away:

- Lowercasing and contraction expansion (e.g. "can't" becomes "cannot").
- A slang dictionary (e.g. "idk" becomes "i do not know").
- Hashtag unwrapping (keeps the word), and normalisation of URLs and user mentions to placeholder tokens.
- Swear-word masking to a single `swear_word` token.
- Repeated-character compression (e.g. "nooooo" becomes "noo").
- Emotional punctuation tokens for repeated `!`, `?`, and `...`.
- Negation binding (e.g. "not happy" becomes "not_happy") and intensifier binding (e.g. "very sad" becomes "very_sad").
- Custom stopword list built from the English stopwords but keeping negations, intensifiers, and modal words that carry emotional meaning.

## Models

Both models use the same shape: a TF-IDF vectoriser (unigrams and bigrams, `min_df=5`) feeding a Logistic Regression classifier with `class_weight='balanced'`.

| Setting | Emotion model | Sentiment model |
| --- | --- | --- |
| TF-IDF max features | 60,000 | 80,000 |
| Solver | liblinear | lbfgs (multinomial) |
| sublinear_tf | off | on |

## Data and Split

- Cleaned dataset: 207,814 single-label records.
- Stratified 80/20 train/test split: 166,251 training and 41,563 test samples.
- Sentiment class balance: neutral 43.4%, positive 32.4%, negative 24.1%.

## Key Results

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Majority-class baseline | 0.266 | (n/a) |
| Emotion (fine-grained) | 0.329 | 0.285 |
| Sentiment (3-class) | 0.650 | 0.647 |

The fine-grained emotion task is hard: with many overlapping, subjective categories the model beats the baseline but stays modest. Collapsing to three sentiment classes roughly doubles accuracy and lifts macro F1 to 0.65, showing that the same features separate broad sentiment far more reliably than subtle emotion.

## Repository Structure

```
.
├── A00057300_Data_Analytics_CW2.ipynb   # full pipeline (cleaning, EDA, both models, evaluation)
├── requirements.txt                     # dependencies
└── README.md                            # this file
```

The notebook expects `goemotions.csv` available locally (in the original it was read from Google Drive; update the `path` variable to point at your copy).

## How to Run

### Google Colab (recommended)

1. Open `A00057300_Data_Analytics_CW2.ipynb` in Colab.
2. Make `goemotions.csv` available (mount Drive or upload it) and set the `path` variable accordingly.
3. Run all cells top to bottom.

### Local

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook A00057300_Data_Analytics_CW2.ipynb
```

## AI Declaration

All modelling, coding, experimentation, and analysis are the author's own work. Generative AI tools were used only for language refinement and to clarify code concepts; the design decisions, interpretations, and conclusions are entirely the author's own.
