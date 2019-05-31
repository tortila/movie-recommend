from parser import Parser, Mode
from predictor import Predictor
import numpy as np
import sys


def main():
    print("-- Welcome to movie-recommend! --")

    # for output readability
    np.set_printoptions(formatter={"float_kind": "{:25f}".format})

    # baseline predictor by default
    mode = Mode.BASELINE

    # read command-line argument, if provided
    if len(sys.argv) > 1:
        if sys.argv[1] == Mode.IMPROVED or sys.argv[1] == Mode.BASELINE:
            mode = sys.argv[1]
            print("    You chose: {} predictor!".format(mode))
        else:
            print(
                "    {} is not a valid argument. Default: {} predictor!".format(
                    sys.argv[1], mode
                )
            )
    else:
        print(
            "    You did not provide any arguments. Default: {} predictor!".format(mode)
        )

    # read and parse text files
    parser = Parser(mode)
    print("    Parser initialized:")
    print(
        "        {} test points and {} training points".format(
            len(parser.test_set), np.count_nonzero(parser.training_matrix)
        )
    )

    # initialize predictor and calculate rmse
    predictor = Predictor(mode, parser.training_matrix, parser.test_set)
    print("    rmse on test data (baseline): {}".format(predictor.rmse_test))
    if predictor.mode == Mode.BASELINE:
        print(
            "    rmse on training data (baseline): {}".format(predictor.rmse_training)
        )
    else:
        print(
            "    rmse on test data (improved): {}".format(predictor.rmse_test_improved)
        )

    # execute histogram plotting and get error distribution
    error_dist = (
        predictor.calculate_absolute_errors(parser.test_set, predictor.improved_matrix)
        if predictor.mode == Mode.IMPROVED
        else predictor.calculate_absolute_errors(
            parser.test_set, predictor.baseline_matrix
        )
    )
    print("    Histogram saved to file. Error distribution: {}".format(error_dist))


if __name__ == "__main__":
    main()
