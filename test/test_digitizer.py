from stitch import Stitch
from svgpathtools import Path, Line

from digitizer import Digitizer
from mock import patch

import pytest


@patch.object(Digitizer, "generate_pattern")
def test_svg_to_pattern(mock_generate_pattern):
    dig = Digitizer()
    dig.filecontents = open("black_rectangle.svg", "r").read()
    dig.svg_to_pattern()
    assert len(dig.all_paths) > 0
    mock_generate_pattern.assert_called_once()


@pytest.mark.parametrize("fill", [(True), (False)])
def test_generate_pattern(fill):
    dig = Digitizer()
    dig.all_paths = [Path(*[Line(start=0, end=100), Line(start=100, end=100+100j),
                            Line(start=100+100j, end=100j), Line(start=100j, end=0)])]
    dig.attributes = [{"fill": "black"}]
    dig.scale = 1.0
    dig.fill = fill
    dig.generate_pattern()
    assert len(dig.pattern.blocks) > 0
    assert len(dig.pattern.blocks[0].stitches) > 0


def test_fill_scan():
    dig = Digitizer()
    dig.fill_color = (0, 0, 0)
    paths = Path(*[Line(start=0, end=100), Line(start=100, end=100+100j),
                   Line(start=100+100j, end=100j), Line(start=100j, end=0)])
    dig.scale = 1.0
    dig.fill = True
    dig.fill_scan(paths)
    assert len(dig.stitches) > 0


@patch.object(Digitizer, "add_block")
def test_switch_color(mock_add_block):
    dig = Digitizer()
    input_stitches = [Stitch(x=0, y=0), Stitch(x=100, y=0), Stitch(x=100, y=100)]
    dig.stitches = input_stitches
    dig.last_color = 0
    new_color = 1
    dig.switch_color(new_color)
    mock_add_block.assert_called_once()
    assert "TRIM" in dig.pattern.blocks[-2].stitches[-1].tags
    assert "COLOR" in dig.pattern.blocks[-1].stitches[-1].tags


def test_add_block():
    dig = Digitizer()
    input_stitches = [Stitch(x=0, y=0), Stitch(x=100, y=0), Stitch(x=100, y=100)]
    dig.stitches = input_stitches
    dig.last_color = (0, 0, 0)
    dig.add_block()
    assert dig.stitches == []
    assert len(dig.pattern.blocks) == 1
    assert dig.pattern.blocks[0].stitches == input_stitches