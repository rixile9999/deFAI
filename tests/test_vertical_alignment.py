import numpy as np
import pytest
from defai.vertical.alignment import align_parties, align_time_series


def test_alignment_reorders_features_and_owner_labels_without_combining_columns():
    data = align_parties({'a': (['b', 'a', 'x'], [[20], [10], [99]]),
                          'b': (['a', 'b'], [[100, 101], [200, 201]])}, labels=[1, 0, 9], label_owner='a')
    np.testing.assert_array_equal(data.features['a'], [[10], [20]])
    np.testing.assert_array_equal(data.features['b'], [[100, 101], [200, 201]])
    np.testing.assert_array_equal(data.labels, [0, 1])
    assert data.size == 2


@pytest.mark.parametrize('ids,x', [(['a', 'a'], [[1], [2]]), (['a'], [[np.nan]]), (['a'], [1])])
def test_bad_local_data_is_rejected(ids, x):
    with pytest.raises(ValueError):
        align_parties({'a': (ids, x), 'b': (['a'], [[1]])})


def test_empty_intersection_and_missing_owner_are_errors():
    with pytest.raises(ValueError, match='No common'):
        align_parties({'a': (['a'], [[1]]), 'b': (['b'], [[2]])})
    with pytest.raises(ValueError, match='together'):
        align_parties({'a': (['a'], [[1]]), 'b': (['a'], [[2]])}, labels=[1])


def test_time_resampling_and_no_extrapolation():
    streams = {'a': ([0, 2], [[0], [4]]), 'b': ([0, 1, 2], [[0], [3], [6]])}
    out = align_time_series(streams, [0, 1, 2])
    np.testing.assert_allclose(out['a'].ravel(), [0, 2, 4])
    with pytest.raises(ValueError, match='extrapolate'):
        align_time_series(streams, [0, 3])
