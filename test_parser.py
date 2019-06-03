import numpy as np
import pytest

from parser import Parser, Mode


@pytest.mark.parametrize("mode", [Mode.BASELINE, Mode.IMPROVED])
def test_parses_test_data(mode, mock_csv_files):
    p = Parser.from_mode(mode)
    assert len(p.test_set) > 0, "number of test points should be greater than 0"


@pytest.mark.parametrize("mode", [Mode.BASELINE, Mode.IMPROVED])
def test_parses_training_data(mode, mock_csv_files):
    p = Parser.from_mode(mode)
    data = p.training_matrix
    assert np.count_nonzero(data), "number of training points should be greater than 0"
