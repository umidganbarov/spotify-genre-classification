# Spotify Genre Classification

Compared KNN and Random Forest on a Spotify dataset to classify music genres.

## Results
| Model | Accuracy |
|-------|----------|
| KNN (k=9) | 29% |
| Random Forest | 79% |

## What I did
- Filtered 8 genres (turkish, rock, metal, folk, opera, etc.)
- Compared KNN vs Random Forest baseline
- Tuned RFC with GridSearchCV (10-fold CV, 300 estimators)

## Stack
Python, Pandas, Scikit-Learn
