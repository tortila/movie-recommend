from Parser import Parser
from Predictor import Predictor
import numpy as np
import sys

BASELINE = "baseline"
IMPROVED = "improved"

def main():
    print "Welcome to movie-recommend"
    # for output readability
    np.set_printoptions(formatter={'float_kind':'{:25f}'.format})
    mode = BASELINE
    if len(sys.argv) > 2:
        if sys.argv[1] == IMPROVED or sys.argv[1] == BASELINE:
            mode = sys.argv[1]
            print "You chose", mode, "predictor!"
        else:
            print sys.argv[1], "is not a valid argument. \nDefault:", mode, "predictor!"
    else:
        print "You did not provide any arguments. \nDefault:", mode, "predictor!"
    parser = Parser(mode)
    predictor = Predictor(mode, parser.training_matrix, parser.test_set)

if __name__ == "__main__":
    main()