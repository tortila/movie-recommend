import pytest

from parser import Parser, Mode
from predictor import Predictor


@pytest.mark.parametrize("mode", [Mode.BASELINE, Mode.IMPROVED])
def test_rmse_calculated(mode, mock_parser_reading_files):
    parser = Parser.from_mode(mode)
    predictor = Predictor(mode, parser.training_matrix, parser.test_set)
    assert predictor.rmse_test > 0.0
    assert predictor.rmse_training > 0.0
    if mode == "improved":
        assert predictor.rmse_test_improved > 0.0
