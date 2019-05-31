import pytest
from hypothesis import given
from hypothesis.strategies import lists, permutations

from parser import Parser, Mode
from predictor import Predictor

movie_titles = "\n".join([f"{i};Random Title {i}" for i in range(1, 6)])


@pytest.fixture(autouse=True)
@given(test_data=lists(permutations(range(1, 4)), min_size=5))
def mock_parser_reading_files(mocker, test_data):
    mocker.patch("parser._data_parse", return_value=test_data)
    mocker.patch("parser._moviename_parse", return_value=movie_titles)


@pytest.mark.parametrize("mode", [Mode.BASELINE, Mode.IMPROVED])
def test_rmse_calculated(mode):
    parser = Parser.from_mode(mode)
    predictor = Predictor(mode, parser.training_matrix, parser.test_set)
    assert predictor.rmse_test > 0.0
    assert predictor.rmse_training > 0.0
    if mode == "improved":
        assert predictor.rmse_test_improved > 0.0
