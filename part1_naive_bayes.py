from pathlib import Path

import numpy as np
import pandas as pd

# Every path starts from the folder that contains this file,
# so the program runs on any computer and from any folder.
pathOg = Path(__file__).parent
test_file = pathOg / "Testing dataset.xlsx"
train_file = pathOg / "Training dataset.xlsx"


def load_dataset(file_path1):
    """Read one Excel file and return (names, attribute names, X, y).

    names = the wine names, attribute names = the 305 column headers,
    X = the 0/1 attribute matrix, y = the labels.

    Columns are taken by position: 0 = wine name, 1 = grade, 2.. = attributes.
    """
    df = pd.read_excel(file_path1)
    names = df.iloc[:, 0].astype(str).tolist()
    attributes = df.columns[2:].tolist ()
    X = df.iloc[:, 2:].to_numpy(dtype=int)  # 305 attributes, each 0 or 1
    y = df.iloc[:, 1].to_numpy(dtype=int)  # 1 = 90+, 0 = below 90
    return names, attributes, X, y

class NaiveBayes:
    """Bernoulli Naive Bayes for 0/1 attributes, with Laplace smoothing."""

    def __init__(self, alpha=1.0):
        """Store the smoothing strength alpha (1.0 = Laplace, add-one)."""
        self.alpha = alpha
        if self.alpha <= 0:
            raise ValueError("alpha must be a positive value")

    def fit(self, X, y):
        """Learn P(class) and P(attribute = 1 | class) from the training data."""
        self.prior = {}
        self.p_present = {}
        self.classes = np.unique(y)  # the labels found in the data, e.g. [0 1]
        for c in self.classes:
            rows = X[y == c]  # the training wines of class c
            self.prior[c] = len(rows) / len(X)
            # Laplace smoothing: (count + alpha) / (n + 2 * alpha). The 2 is for
            # the two outcomes (0 or 1). A word never seen in a class gets a small
            # probability instead of 0; one such word in a test wine would
            # otherwise make that class's whole product 0.
            self.p_present[c] = (rows.sum(axis=0) + self.alpha) / (
                len(rows) + 2 * self.alpha
            )
        return self

    def log_posterior(self, X):
        """Return log P(class) + log P(wine | class) for each wine and class.

        This joint log score differs from the true log posterior only by
        log P(wine), which is the same for both classes, so the winner is the same.
        Logs turn the product of 305 probabilities into a sum. That is standard
        practice and numerically safer, but this data does not need it: most
        factors are close to 1, the smallest product is about 1e-22 on the
        testing wines and 4e-29 on the training wines (a float goes down to
        about 1e-308), and the predictions are the same without logs.
        """
        scores = []
        for c in self.classes:
            p = self.p_present[c]
            # present words add log(p); absent words add log(1 - p)
            log_score = X @ np.log(p) + (1 - X) @ np.log(1 - p)
            scores.append(np.log(self.prior[c]) + log_score)
        return np.column_stack(scores)  # shape: (number of wines, number of classes)

    def predict(self, X):
        """Pick the class with the highest score for each wine."""
        return self.classes[np.argmax(self.log_posterior(X), axis=1)]


def accuracy(y_true, y_pred):
    """Return the share of wines predicted correctly, as a fraction of 1.

    Part 2 of the project: #correctly predicted wine / #total wine.
    The caller prints this with the .2% format, which multiplies by 100
    itself, so this must return 0.9474 and not 94.74.
    """
    # y_true == y_pred compares the two arrays position by position and gives
    # an array of True/False. Summing it counts the True values, because
    # Python counts True as 1 and False as 0. A single / keeps the decimals.
    return (y_true == y_pred).sum() / len(y_true)


def write_table(path, y_true, y_pred):
    """Write the 'Real grade / Predicted grade' table Dr. Chen asked for."""
    with open(path, "w") as f:
        f.write(f"{'Real grade':<15}{'Predicted grade'}\n")
        f.writelines(f"{true:<15}{pred}\n" for true, pred in zip(y_true, y_pred))


def write_excel(path, test_file, y_true, y_pred):
    """Copy the testing table to Excel and put the results table below it."""
    test_df = pd.read_excel(test_file)
    test_df = test_df.rename(columns={"Unnamed: 0": "Name of Wine"})
    results_df = pd.DataFrame({"Real grade": y_true, "Predicted grade": y_pred})
    with pd.ExcelWriter(path) as writer:
        test_df.to_excel(writer, sheet_name="Results", index=False)
        start = len(test_df) + 3  # header + 19 wines, then 2 empty rows
        results_df.to_excel(writer, sheet_name="Results", startrow=start, index=False)


def save_failed(path):
    """Tell the user which result file could not be written, and the likely reason.

    PermissionError means the computer refused to write the file: the file or
    its folder is read-only, or another program has locked the file (Excel does
    this on Windows; on a Mac its lock does not block Python).
    """
    print(
        f"Could not save {path.name} - the file or its folder is read-only, "
        "or the file is open in another program. Fix that and run again."
    )


def main():
    """Train on the training set, predict the testing set, report and save."""
    _, train_attributes, X, y = load_dataset(train_file)  # _ = wine names, not needed here
    _, test_attributes, X_t, y_t = load_dataset(test_file)
    if train_attributes != test_attributes:
        raise ValueError("Training and testing attributes do not match")
    nb = NaiveBayes().fit(X, y)  # Part 1: learn from the training data only
    predictions = nb.predict(X_t)

    # Part 2: accuracy = #correct / #total, printed before saving any file
    correct = (y_t == predictions).sum()
    print(
        f"Naive Bayes accuracy: {correct}/{len(y_t)} = "
        f"{accuracy(y_t, predictions):.2%}"
    )
    # Each file has its own try block, so one failed file does not stop the other.
    txt_path = pathOg / "results.txt"
    try:
        write_table(txt_path, y_t, predictions)
    except PermissionError:
        save_failed(txt_path)
    xlsx_path = pathOg / "results.xlsx"
    try:
        write_excel(xlsx_path, test_file, y_t, predictions)
    except PermissionError:
        save_failed(xlsx_path)


if __name__ == "__main__":
    main()
