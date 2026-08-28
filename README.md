# Spotify Genre Classification

Compares four classification models to predict a track's genre from Spotify audio features, across 8 genres: `acoustic`, `classical`, `electro`, `electronic`, `guitar`, `k-pop`, `study`, and `turkish`.

## Results

| Model | Test Accuracy | 10-Fold CV Score |
|---|---|---|
| KNN (k=7) | 61.70% | 61.06% |
| Logistic Regression | 58.10% | 57.79% |
| Decision Tree (max_depth=5) | 65.00% | 59.56% |
| **Random Forest** | **78.30%** | **73.96%** |

Random Forest was the best performer, so a full classification report was generated for it:

| Genre | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| acoustic | 0.67 | 0.72 | 0.69 | 232 |
| classical | 0.94 | 0.92 | 0.93 | 236 |
| electro | 0.78 | 0.77 | 0.78 | 255 |
| electronic | 0.67 | 0.64 | 0.66 | 247 |
| guitar | 0.79 | 0.78 | 0.78 | 260 |
| k-pop | 0.75 | 0.75 | 0.75 | 251 |
| study | 0.88 | 0.92 | 0.90 | 250 |
| turkish | 0.79 | 0.77 | 0.78 | 269 |
| **accuracy** | | | **0.78** | 2000 |

Classical and study are the easiest genres to separate — they have the most distinctive audio profiles. Electronic and electro are the hardest pair to tell apart, which tracks with how sonically similar those two genres are.

## What I did

- Loaded the Spotify dataset and dropped non-predictive identifier columns (`n`, `track_id`, `artists`, `album_name`, `track_name`)
- Filtered the dataset down to 8 genres
- Converted the `explicit` column to `int`
- Built 4 pipelines (`StandardScaler` + classifier): KNN, Logistic Regression, Decision Tree, Random Forest
- Evaluated each model on a held-out test split and with 10-fold cross-validation
- Generated a classification report for the best model and a bar chart comparing test accuracy vs. CV score across all 4 models

## Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib
