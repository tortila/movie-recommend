import pytest

from dataset import Dataset
from predictor import Predictor, Mode


@pytest.mark.parametrize("mode", [Mode.BASELINE, Mode.IMPROVED])
def test_rmse_calculated(mode, mock_reading_dataset_from_files):
    d = Dataset.from_file("whatever, it will be mocked anyway")
    training_matrix = d.as_umr_matrix()
    test_set = d.records
    predictor = Predictor.from_mode(mode).build(training_matrix, test_set)
    assert predictor.rmse_test >= 0.0
    assert predictor.rmse_training >= 0.0
    if mode == "improved":
        assert predictor.rmse_test_improved >= 0.0
