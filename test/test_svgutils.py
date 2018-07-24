from svgpathtools import Path, Line
from svgutils import stack_paths, scan_lines


def test_stack_paths():
    all_paths = [Path(*[Line(start=0, end=100), Line(start=100, end=100 + 100j),
                        Line(start=100 + 100j, end=100j), Line(start=100j, end=0)])]
    attributes = [{"fill": "black"}]

    all_paths_new, attributes_new = stack_paths(all_paths, attributes)
    assert len(all_paths) == len(all_paths_new)
    assert len(attributes_new) == len(attributes)


def test_stack_paths2():
    blo = Path(*[Line(start=0, end=100), Line(start=100, end=100 + 100j),
                 Line(start=100 + 100j, end=100j), Line(start=100j, end=0)])
    all_paths = [blo, blo.translated(110)]
    attributes = [{"fill": "black"}, {"fill": "black"}]

    all_paths_new, attributes_new = stack_paths(all_paths, attributes)
    assert all_paths == all_paths_new
    assert attributes_new == attributes


def test_scan_lines():
    paths = Path(*[Line(start=0, end=100), Line(start=100, end=100 + 100j),
                   Line(start=100 + 100j, end=100j), Line(start=100j, end=0)])
    lines = scan_lines(paths)
    assert len(lines) > 0