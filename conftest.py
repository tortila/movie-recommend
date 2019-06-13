from pytest import fixture
from hypothesis import given, strategies as st
from hypothesis_csv.strategies import csv as csv_st

from dataset import Dataset

integers_strategy = st.integers(min_value=1, max_value=5)
_ratings = "ratings.csv"


@given(
    csv_data_small_batch=csv_st(
        columns=[integers_strategy, integers_strategy, integers_strategy], lines=25
    )
)
def fake_data_files(csv_data_small_batch, files_path):
    test_data = files_path / _ratings
    test_data.write_text(csv_data_small_batch)


@fixture
def get_mocked_csv_path(tmp_path):
    fake_data_files(files_path=tmp_path)
    return tmp_path / _ratings


@fixture
@given(
    test_data=st.lists(
        st.tuples(integers_strategy, integers_strategy, integers_strategy),
        min_size=10,
        max_size=15,
    )
)
def mock_reading_dataset_from_files(mocker, test_data):
    mocker.patch("dataset.Dataset.from_file", return_value=Dataset(records=test_data))
