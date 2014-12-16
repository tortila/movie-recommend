from Parser import Parser
from Predictor import Predictor

BASELINE = "oligo854"
IMPROVED = "improved"

def main():
    print "Welcome to movie-recommend"
    parser = Parser(BASELINE)
    predictor = Predictor(parser.umr, parser.users_number, parser.movies_number)

if __name__ == "__main__":
    main()