"""CS 6610 Project 1, Part 2: 5-fold cross-validation of Naive Bayes.

Author: Loc Nguyen
Reuses load_dataset and NaiveBayes from part1_naive_bayes.py (Part 1) on all 1010
wines in Full Wine Data.xlsx. Each of the 5 folds is the testing set once while the other
4 folds train a fresh model; TP, FP, TN and FN are counted per fold (positive
class = 90+) and averaged over the 5 folds.

Run:     python part2_cross_validation.py
Output:  the table on screen, plus results_cv.txt and results_cv.xlsx next to this file
Needs:   part1_naive_bayes.py and Full Wine Data.xlsx in the same folder;
         numpy, pandas, openpyxl
"""
from pathlib import Path

import numpy as np
import pandas as pd

from part1_naive_bayes import NaiveBayes, load_dataset, save_failed

pathOg = Path(__file__).parent
file_path = pathOg / "Full Wine Data.xlsx"


def fold_calculator(y, k=5, seed=0):
    """Split the row numbers into k shuffled, stratified folds.

    Stratified = every fold gets the same share of 90+ wines. The file lists all
    90+ wines first, so without shuffling fold 1 would hold only 90+ wines.
    k is the number of folds; the fixed seed gives the same folds on every run.
    Returns a list of k arrays of row numbers; every row is in exactly one fold.
    """
    # Random number generator with a fixed seed, for reproducibility
    rng = np.random.default_rng(seed)
    # np.where finds the row numbers of the 90+ wines; permutation shuffles them
    pos = rng.permutation(np.where(y == 1)[0])
    # the same for the below-90 wines
    neg = rng.permutation(np.where(y == 0)[0])
    # cut each shuffled list into k parts (sizes differ by at most 1)
    pos_parts = np.array_split(pos, k)
    neg_parts = np.array_split(neg, k)
    folds = []
    for i in range(k):
        # fold i = part i of the 90+ wines + part i of the below-90 wines
        folds.append(np.concatenate([pos_parts[i], neg_parts[i]]))
    return folds


def confusion_counts(y_true, y_pred):
    """Return (TP, FP, TN, FN), with 90+ (label 1) as the positive class."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())  # real 90+, predicted 90+
    fp = int(((y_true == 0) & (y_pred == 1)).sum())  # real <90, predicted 90+
    tn = int(((y_true == 0) & (y_pred == 0)).sum())  # real <90, predicted <90
    fn = int(((y_true == 1) & (y_pred == 0)).sum())  # real 90+, predicted <90
    return tp, fp, tn, fn


def main():
    """Run 5-fold cross-validation, then print and save the TP/FP/TN/FN table."""
    _, attributes, X, y = load_dataset(file_path)
    print(f"Wines: {len(X)}")
    print(f"Attributes: {len(attributes)}")
    print(f"90+ wines: {sum(y)}")

    folds = fold_calculator(y)

    row = []
    for i in range(len(folds)):
        test_idx = folds[i] # fold i is the testing set
                            # the other 4 folds, joined, are the training set
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        model = NaiveBayes().fit(X[train_idx], y[train_idx])  # a fresh model
        prediction = model.predict(X[test_idx])
        tp, fp, tn, fn = confusion_counts(y[test_idx], prediction)
        row.append((tp, fp, tn, fn))

    table = pd.DataFrame(
        row,
        columns=["TP", "FP", "TN", "FN"],
        index=[f"Fold {i + 1}" for i in range(len(row))],
    )
    # accuracy of each fold = (TP + TN) / fold size; the result is their average
    accuracy = ((table["TP"] + table["TN"]) / table.sum(axis=1)).mean()
    table.loc["Average on all"] = table.mean()
    print(table )
    print(f"Average accuracy: {accuracy:.2%}")

    # Each file has its own try block, so one failed file does not stop the other.
    txt_path = pathOg / "results_cv.txt"
    try:
        with open(txt_path, "w") as f:
            f.write("Results 5-fold cross-validation\n\n")
            f.write(table.to_string() + "\n\n")
            f.write(f"Average accuracy: {accuracy:.2%}\n")
    except PermissionError:
        save_failed(txt_path)
    xlsx_path = pathOg / "results_cv.xlsx"
    try:
        table.to_excel(xlsx_path, sheet_name="5-fold cross-validation")
    except PermissionError:
        save_failed(xlsx_path)


if __name__ == "__main__":
    main()
