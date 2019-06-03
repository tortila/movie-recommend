from parser import Parser, Mode
from predictor import Predictor
import numpy as np
import sys
import logging

logging.basicConfig(
    format="%(asctime)s %(module)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    logging.info("-- Welcome to movie-recommend! --")

    # for output readability
    np.set_printoptions(formatter={"float_kind": "{:25f}".format})

    # baseline predictor by default
    mode = Mode.BASELINE

    # read command-line argument, if provided
    if len(sys.argv) > 1:
        if sys.argv[1] == Mode.IMPROVED or sys.argv[1] == Mode.BASELINE:
            mode = sys.argv[1]
            logging.info("You chose: %s predictor!", mode)
        else:
            logging.warning(
                "%s is not a valid argument. Default: %s predictor!", sys.argv[1], mode
            )
    else:
        logging.warning(
            "You did not provide any arguments. Default: %s predictor!", mode
        )

    # read and parse text files
    parser = Parser.from_mode(mode)
    training_matrix = parser.training_matrix
    test_matrix = parser.test_set
    logging.info(
        "Parser initialized with: %d test points, %d training points",
        len(test_matrix),
        np.count_nonzero(training_matrix),
    )

    # initialize predictor and calculate rmse
    predictor = Predictor(mode, training_matrix, test_matrix)
    logging.info("rmse on test data (baseline): %f", predictor.rmse_test)
    if predictor.mode == Mode.BASELINE:
        logging.info("rmse on training data (baseline): %f", predictor.rmse_training)
    else:
        logging.info("rmse on test data (improved): %f", predictor.rmse_test_improved)

    # execute histogram plotting and get error distribution
    error_dist = (
        predictor.calculate_absolute_errors(parser.test_set, predictor.improved_matrix)
        if predictor.mode == Mode.IMPROVED
        else predictor.calculate_absolute_errors(
            parser.test_set, predictor.baseline_matrix
        )
    )
    logging.info("Histogram saved to file. Error distribution: %s", error_dist)


if __name__ == "__main__":
    main()
