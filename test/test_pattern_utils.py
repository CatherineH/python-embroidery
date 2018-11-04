from xml.etree.ElementTree import parse

import pytest
from block import Block
from brother import pattern_to_csv
from pattern import Pattern
from pattern_utils import NextAvailableCone, pattern_to_svg
from stitch import Stitch


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


def test_pattern_to_svg():
    filename = "test_pattern_to_svg.svg"
    stitches = [Stitch(x=0, y=0, tags=["JUMP"]),
                Stitch(x=0, y=100, tags=["STITCH"]),
                Stitch(x=100, y=100, tags=["STITCH"]),
                Stitch(x=100, y=0, tags=["STITCH"]),
                Stitch(x=0, y=0, tags=["STITCH"])]
    pattern = Pattern()
    block = Block(stitches=stitches)
    pattern.add_block(block)
    pattern_to_svg(pattern, filename=filename)
    root = parse(filename).getroot()

    filename2 = "test_pattern_to_svg.csv"
    pattern_to_csv(pattern, filename=filename2)
    assert False
