import pytest

from dataset import Dataset
from predictor import Mode


@pytest.mark.parametrize("mode", [Mode.BASELINE, Mode.IMPROVED])
def test_parses_test_data(mode, get_mocked_csv_path):
    d = Dataset.from_file(get_mocked_csv_path)
    assert len(d.records) > 0, "number of test points should be greater than 0"
