import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis_csv.strategies import csv as csv_st

from parser import Parser, Mode


@pytest.fixture(autouse=True)
@given(
    csv_test=csv_st(
        columns=[
            st.integers(min_value=1, max_value=5),
            st.integers(min_value=1, max_value=5),
            st.integers(min_value=1, max_value=5),
        ],
        lines=20,
    ),
    csv_training=csv_st(
        columns=[
            st.integers(min_value=1, max_value=20),
            st.integers(min_value=1, max_value=10),
            st.integers(min_value=1, max_value=5),
        ],
        lines=100,
    ),
)
def fake_data(csv_training, csv_test, mocker, tmp_path):
    test_data_baseline = tmp_path / f"baseline.test"
    test_data_improved = tmp_path / f"improved.test"
    test_data_baseline.write_text(csv_test)
    test_data_improved.write_text(csv_test)
    training_data_baseline = tmp_path / f"baseline.training"
    training_data_improved = tmp_path / f"improved.training"
    training_data_baseline.write_text(csv_training)
    training_data_improved.write_text(csv_training)
    movie_data_baseline = tmp_path / f"baseline.moviename"
    movie_data_improved = tmp_path / f"improved.moviename"
    movie_data_baseline.write_text(
        "\n".join([f"{i};Random Title {i}" for i in range(1, 21)])
    )
    movie_data_improved.write_text(
        "\n".join([f"{i};Random Title {i}" for i in range(1, 21)])
    )
    mocker.patch("parser.DIR", tmp_path)


@pytest.mark.parametrize("mode", [Mode.BASELINE, Mode.IMPROVED])
def test_parses_test_data(mode):
    p = Parser.from_mode(mode)
    assert len(p.test_set) > 0, "number of test points should be greater than 0"


@pytest.mark.parametrize("mode", [Mode.BASELINE, Mode.IMPROVED])
def test_parses_training_data(mode):
    p = Parser.from_mode(mode)
    data = p.training_matrix
    assert np.count_nonzero(data), "number of training points should be greater than 0"
