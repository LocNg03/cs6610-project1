# CS 6610 Project 1 — Predicting 90+ wines with Naive Bayes

Course project for CS 6610 (Data Mining), Troy University. Each wine is one row of
305 binary attributes from the Computational Wine Wheel (1 = the review mentions that
word or phrase); the label is 1 for 90+ points and 0 for below 90.

## Files

| File | What it does |
|---|---|
| `part1_naive_bayes.py` | Part 1: Bernoulli Naive Bayes (Laplace smoothing, log scores) trained on `Training dataset.xlsx` (200 wines) and tested on `Testing dataset.xlsx` (19 wines). Writes `results.txt` and `results.xlsx`. |
| `part2_cross_validation.py` | Part 2: stratified 5-fold cross-validation of the same classifier on `Full Wine Data.xlsx` (1010 wines); counts TP, FP, TN, FN per fold (positive = 90+) and averages them. Writes `results_cv.txt` and `results_cv.xlsx`. |

## Run

```bash
pip install numpy pandas openpyxl
python part1_naive_bayes.py
python part2_cross_validation.py
```

## Results

Part 1 (fixed train/test split): 18/19 correct = 94.74%.

Part 2 (5-fold cross-validation, seed 0):

| | TP | FP | TN | FN |
|---|---|---|---|---|
| Fold 1 | 80 | 12 | 88 | 23 |
| Fold 2 | 91 | 2 | 98 | 11 |
| Fold 3 | 87 | 1 | 99 | 15 |
| Fold 4 | 86 | 8 | 92 | 16 |
| Fold 5 | 81 | 13 | 86 | 21 |
| Average on all | 85.0 | 7.2 | 92.6 | 17.2 |

Average accuracy: 87.92%.

Author: Loc Nguyen
