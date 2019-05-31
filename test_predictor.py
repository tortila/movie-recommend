import pytest

from parser import Parser
from predictor import Predictor

test_data = [
    [1, 1, 1],
    [2, 1, 3],
    [2, 2, 4],
    [3, 1, 1],
    [3, 2, 5],
]

movie_titles = [
    [1, "Shrek"],
    [2, "Matrix"]
]


@pytest.fixture(autouse=True)
def mock_parser_reading_files(mocker):
    mocker.patch("parser._data_parse", return_value=test_data)
    mocker.patch("parser._moviename_parse", return_value=movie_titles)


@pytest.mark.parametrize("mode", ["baseline", "improved"])
def test_rmse_calculated(mode):
    parser = Parser(mode)
    predictor = Predictor(mode, parser.training_matrix, parser.test_set)
    assert predictor.rmse_test > 0.0
    assert predictor.rmse_training > 0.0
    if mode == "improved":
        assert predictor.rmse_test_improved > 0.0
