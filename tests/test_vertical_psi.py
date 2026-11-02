import numpy as np
import pytest
from defai.vertical.alignment import align_parties
from defai.vertical.psi import DHParty, P, Q, intersection_rows


def test_three_party_intersection_reorders_local_rows_consistently():
    keys = {'a': ['one', 'two', 'three'], 'b': ['three', 'other', 'one'], 'c': ['one', 'three']}
    rows = intersection_rows(keys)
    recovered = {k: [keys[k][i] for i in r] for k, r in rows.items()}
    assert recovered['a'] == recovered['b'] == recovered['c']
    assert set(recovered['a']) == {'one', 'three'}
    assert P.bit_length() == 2048 and P == 2 * Q + 1


def test_psi_protocol_payload_is_group_elements_not_identifiers():
    party = DHParty(['private-customer'], b'session')
    encoded = party.encode()
    assert len(encoded) == 1 and isinstance(encoded[0], int)
    assert party.blind(encoded) != encoded
    with pytest.raises(ValueError, match='subgroup'):
        party.blind([1])


def test_dh_alignment_keeps_labels_with_the_owner():
    data = align_parties({'a': (['b', 'a'], [[2], [1]]), 'b': (['a', 'b'], [[10], [20]])},
                         labels=[200, 100], label_owner='a', mode='dh-psi')
    np.testing.assert_array_equal(data.features['a'].ravel() * 10, data.features['b'].ravel())
    np.testing.assert_array_equal(data.features['a'].ravel() * 100, data.labels)
