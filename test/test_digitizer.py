from configure import MINIMUM_STITCH_LENGTH, MINIMUM_STITCH_DISTANCE
from os.path import join
from pattern_utils import pattern_to_svg
from stitch import Stitch
from svgpathtools import Path, Line

from digitizer import Digitizer
from mock import patch

import pytest
from svgwrite import Drawing



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


@pytest.mark.parametrize("fill", [(True), (False)])
def test_generate_pattern_colors(fill):
    dig = Digitizer()
    diff = 110
    blo = Path(*[Line(start=0, end=100), Line(start=100, end=100 + 100j),
                 Line(start=100 + 100j, end=100j), Line(start=100j, end=0)])
    dig.all_paths = [blo, blo.translated(diff)]
    dig.attributes = [{"fill": "black"}, {"fill": "red"}]
    dig.scale = 1.0
    dig.fill = fill
    dig.generate_pattern()

    print([(i, block.stitches[0].tags) for i, block in enumerate(dig.pattern.blocks)])
    assert len(dig.pattern.thread_colors) == 2
    assert len(dig.pattern.blocks[0].stitches) > 0
    assert "JUMP" in dig.pattern.blocks[0].stitches[0].tags
    assert "STITCH" in dig.pattern.blocks[1].stitches[0].tags
    assert "TRIM" in dig.pattern.blocks[2].stitches[0].tags
    assert "COLOR" in dig.pattern.blocks[3].stitches[0].tags
    assert "STITCH" in dig.pattern.blocks[4].stitches[0].tags
    assert "END" in dig.pattern.blocks[5].stitches[0].tags
    assert len(dig.pattern.blocks[4].stitches) > 0



def test_fill_scan():
    dig = Digitizer()
    dig.fill_color = (0, 0, 0)
    paths = Path(*[Line(start=0, end=100), Line(start=100, end=100+100j),
                   Line(start=100+100j, end=100j), Line(start=100j, end=0)])
    dig.scale = 1.0
    dig.fill = True
    dig.fill_scan(paths)
    assert len(dig.stitches) > 0


def test_switch_color():
    dig = Digitizer()
    dig.last_stitch = Stitch(["TRIM"], 0, 0)
    input_stitches = [Stitch(x=0, y=0), Stitch(x=100, y=0), Stitch(x=100, y=100)]
    dig.stitches = input_stitches
    dig.last_color = 0
    new_color = 1
    dig.switch_color(new_color)
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


def test_generate_stroke_width():
    dig = Digitizer()
    paths = [Path(*[Line(start=0, end=100), Line(start=100, end=100+100j),
                            Line(start=100+100j, end=100j), Line(start=100j, end=0)])]
    new_paths = dig.generate_stroke_width(paths, 3*MINIMUM_STITCH_DISTANCE)
    assert len(new_paths) == len(paths)*3


def test_generate_straight_stroke():
    dig = Digitizer()
    paths = [Path(*[Line(start=0, end=100), Line(start=100, end=100 + 100j),
                    Line(start=100 + 100j, end=100j), Line(start=100j, end=0)])]
    dig.stroke_color = (0, 0, 0)
    dig.scale = 1.0
    dig.generate_straight_stroke(paths)
    assert len(dig.stitches) > len(paths)


def test_jump_reduction():
    paths = []
    rect_width = 100
    rect_height = rect_width / 2
    for i in range(3):
        y_offset = rect_width*i*1j
        corners = [rect_height, rect_width+rect_height,
                   rect_width+rect_height + rect_height*1j,
                   rect_height*1j+ rect_height]
        corners = [c+y_offset for c in corners]
        lines = [Line(start=corners[j], end=corners[(j+1) % len(corners)])
                 for j in range(len(corners))]
        _path = Path(*lines)
        _path = _path.rotated(i*20)
        paths += list(_path)

    max_y = max([p.start.imag for p in paths]+[p.end.imag for p in paths])
    max_x = max([p.start.real for p in paths]+[p.end.real for p in paths])
    filename = "test_jump_reduction.svg"
    viewbox = [0, -rect_height, max_x+2*rect_height, max_y+2*rect_height]
    dwg = Drawing(filename, width="10cm",
                  viewBox=" ".join([str(b) for b in viewbox]))
    dwg.add(dwg.path(d=Path(*paths).d()))
    dwg.save()
    dig = Digitizer()
    dig.filecontents = open(filename, "r").read()
    dig.svg_to_pattern()
    pattern_to_svg(dig.pattern, join(filename + ".svg"))




