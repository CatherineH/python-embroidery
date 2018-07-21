from svgpathtools import Path, Line
from svgutils import stack_paths


def test_stack_paths():
    all_paths = [Path(*[Line(start=0, end=100), Line(start=100, end=100 + 100j),
                        Line(start=100 + 100j, end=100j), Line(start=100j, end=0)])]
    attributes = [{"fill": "black"}]

    all_paths_new, attributes_new = stack_paths(all_paths, attributes)
    assert len(all_paths) == len(all_paths_new)
    assert len(attributes_new) == len(attributes)