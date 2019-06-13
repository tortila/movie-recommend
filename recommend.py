from pathlib import Path

from dataset import Dataset
from predictor import BaselinePredictor, ImprovedPredictor, Mode
import numpy as np
import sys
import logging

import matplotlib

# a workaround to avoid depending on _tkinter package (the default "tk" backend is not used anyway)
matplotlib.use("agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s %(module)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# to be called from main program
def calculate_absolute_errors(test_set, source, mode):
    filename = f"abs_errors_{mode.value}.png"
    # plot a histogram
    hist_data = [
        (abs(test_set[i][2] - source[test_set[i][0] - 1, test_set[i][1] - 1]))
        for i in range(len(test_set))
    ]
    hist, bins = np.histogram(hist_data, bins=range(10))
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align="center", width=0.7)
    plt.xlabel("Absolute error")
    plt.ylabel("Count")
    plt.title(
        "Histogram of the distribution of the absolute errors for "
        + mode.value
        + " predictor\n"
    )
    plt.grid(True)
    plt.savefig(filename)
    return [x for x in hist if x > 0]


def main():
    logging.info("-- Welcome to movie-recommend! --")

    # for output readability
    np.set_printoptions(formatter={"float_kind": "{:25f}".format})

    # baseline predictor by default
    mode = Mode.BASELINE

    # read command-line argument, if provided
    if len(sys.argv) > 1:
        if sys.argv[1] in {Mode.IMPROVED.value, Mode.BASELINE.value}:
            mode = Mode(sys.argv[1])
            logging.info("You chose: %s predictor!", mode.value)
        else:
            logging.warning(
                "%s is not a valid argument. Default: %s predictor!",
                sys.argv[1],
                mode.value,
            )
    else:
        logging.warning(
            "You did not provide any arguments. Default: %s predictor!", mode.value
        )

    # read and parse text files
    training_matrix = Dataset.from_file(
        Path(f"data/{mode.value}.training")
    ).as_umr_matrix()
    test_records = Dataset.from_file(Path(f"data/{mode.value}.test")).records
    logging.info(
        "Parser initialized with: %d test points, %d training points",
        len(test_records),
        np.count_nonzero(training_matrix),
    )

    if mode == Mode.BASELINE:
        predictor = BaselinePredictor(training_matrix, test_records)
        logging.info("rmse on training data (baseline): %f", predictor.rmse_training)
        error_dist = calculate_absolute_errors(
            test_records, predictor.baseline_matrix, mode
        )
    elif mode == Mode.IMPROVED:
        predictor = ImprovedPredictor(training_matrix, test_records)
        logging.info("rmse on test data (improved): %f", predictor.rmse_test_improved)
        error_dist = calculate_absolute_errors(
            test_records, predictor.improved_matrix, mode
        )

    logging.info("RMSE on test data (baseline): %f", predictor.rmse_test)
    logging.info("Histogram saved to file. Error distribution: %s", error_dist)


if __name__ == "__main__":
    main()
