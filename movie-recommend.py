from Parser import Parser
from Predictor import Predictor
import numpy as np

BASELINE = "oligo854"
IMPROVED = "improved"

def main():
    print "Welcome to movie-recommend"
    # for output readability
    np.set_printoptions(formatter={'float_kind':'{:25f}'.format})
    parser = Parser(BASELINE)
    predictor = Predictor(parser.training_matrix, parser.test_set)

if __name__ == "__main__":
    main()