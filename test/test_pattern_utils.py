import pytest
from pattern_utils import NextAvailableCone


@pytest.mark.parametrize('test_data',
                          [{'name':'None as last', 'i': 0, 'j': 0, 'last_i': None,
                            'last_j': None, 'next_i': 1, 'next_j': 0},
                   {'name': '-x', 'i': 0, 'j': 0, 'last_i': -1, 'last_j': 0, 'next_i': 1,
                    'next_j': 0},
                           {'name': '+x', 'i': 0, 'j': 0, 'last_i': 1, 'last_j': 0,
                            'next_i': -1, 'next_j': 0},
                           {'name': '-y', 'i': 0, 'j': 0, 'last_i': 0, 'last_j': -1,
                            'next_i': 0, 'next_j': 1},
                           {'name': '+y', 'i': 0, 'j': 0, 'last_i': 0, 'last_j': 1,
                            'next_i': 0, 'next_j': -1},
                   ])
def test_next_cone(test_data):
    for next_i, next_j in NextAvailableCone(test_data['i'], test_data['j'],
                                            test_data['last_i'], test_data['last_j']):
        assert next_i == test_data['next_i']
        assert next_j == test_data['next_j']
        break