from parser import Parser
from predictor import Predictor
import numpy as np
import sys

BASELINE = "baseline"
IMPROVED = "improved"


def main():
    print("-- Welcome to movie-recommend! --")

    # for output readability
    np.set_printoptions(formatter={"float_kind": "{:25f}".format})

    # baseline predictor by default
    mode = BASELINE

    # read command-line argument, if provided
    if len(sys.argv) > 1:
        if sys.argv[1] == IMPROVED or sys.argv[1] == BASELINE:
            mode = sys.argv[1]
            print("\tYou chose: %s predictor!", mode)
        else:
            print(
                "\t %s is not a valid argument. Default: %s predictor!",
                sys.argv[1],
                mode,
            )
    else:
        print("\tYou did not provide any arguments. Default:"), mode, "predictor!"

    # read and parse text files
    parser = Parser(mode)
    print("\tParser initialized:")
    print(
        "\t\t %d test points and %dtraining points",
        len(parser.test_set),
        np.count_nonzero(parser.training_matrix),
    )

    # initialize predictor and calculate rmse
    predictor = Predictor(mode, parser.training_matrix, parser.test_set)
    print("\trmse on test data (baseline): %s", predictor.rmse_test)
    if predictor.mode == BASELINE:
        print("\trmse on training data (baseline): %s", predictor.rmse_training)
    else:
        print("\trmse on test data (improved): %s", predictor.rmse_test_improved)

    # execute histogram plotting and get error distribution
    error_dist = (
        predictor.calculate_absolute_errors(parser.test_set, predictor.improved_matrix)
        if predictor.mode == IMPROVED
        else predictor.calculate_absolute_errors(
            parser.test_set, predictor.baseline_matrix
        )
    )
    print("\tHistogram saved to file. Error distribution: %s", error_dist)


if __name__ == "__main__":
    main()
