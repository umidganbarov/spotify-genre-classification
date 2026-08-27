# Spotify Genre Classification

Compared KNN and Random Forest on a Spotify dataset to classify music genres across 8 genres (turkish, rock, metal, folk, opera, brazil, study, rock-n-roll).

## Results
| Model | Accuracy |
|-------|----------|
| KNN (k=9) | 30% |
| Random Forest | 78.35% |
| RFC + GridSearchCV | 79.1% |

## What I did
- Filtered 8 genres from a large Spotify dataset
- Compared KNN vs Random Forest baseline
- Tuned RFC with GridSearchCV (10-fold CV, 11,520 fits)
- GridSearchCV best params: `bootstrap=False, max_depth=20, max_features='sqrt', n_estimators=300`

## Stack
Python, Pandas, NumPy, Scikit-Learn
