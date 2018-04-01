from collections import defaultdict
from math import asin
from os.path import join, dirname
from time import time
from xml.dom.minidom import parseString

import webcolors
css2_names_to_hex = webcolors.css2_names_to_hex
from webcolors import hex_to_rgb, rgb_to_hex

try:
    # potrace is wrapped in a try/except statement because the digitizer might sometimes
    # be run on an environment where Ctypes are not allowed
    import potrace
    from potrace import BezierSegment, CornerSegment
except:
    potrace = None
    BezierSegment = None
    CornerSegment = None

from svgwrite.shapes import Circle
from configure import minimum_stitch, maximum_stitch, DEBUG
from svgpathtools import svgdoc2paths, Line, Path, parse_path
import matplotlib.pyplot as plt
from brother import Pattern, Stitch, Block, BrotherEmbroideryFile, pattern_to_csv, \
    pattern_to_svg, nearest_color

import svgwrite
from numpy import argmax, pi, ceil, sign


def project(A, B, path):
    # given a point A on path, and point B within the path, return point C that this line
    # projects on to
    pathsize = path.bbox()
    pathsize = ((pathsize[1]-pathsize[0])**2+(pathsize[3]-pathsize[2])**2)**0.5
    m = (B.imag-A.imag)/(B.real-A.real)
    b = A.imag-m*A.real
    # right
    if B.real >= A.real:
        end = pathsize+(m*pathsize+b)*1j
    elif B.real < A.real: # left
        end = -pathsize + (-m*pathsize+b)*1j
    intersections = path.intersect(Line(start=B, end=end))
    intersections = [x[0][1].point(x[0][2]) for x in intersections]
    # find the closest point to B
    intersections.sort(key=lambda x: abs(x-B))
    return intersections[0]


def perpendicular(A, B, path):
    # make a line that intersects A, that is perpendicular to the line AB, then return
    # the points at which it intersects the path
    m = (B.imag-A.imag)/(B.real-A.real)
    mp = -1.0/m
    bp = A.imag - mp * A.real
    b = A.imag -m * A.real
    test_line = Line(start=path.bbox()[0]+(path.bbox()[0]*mp+bp)*1j,
                     end=path.bbox()[1]+(path.bbox()[1]*mp+bp)*1j)
    intersections = path.intersect(test_line)
    # this is not guaranteed to work... need some way of looking forward and backward
    intersections = [x[0][1].point(x[0][2]) for x in intersections]
    upper_intersections = [x for x in intersections if x.imag >= m*x.real+b]
    lower_intersections = [x for x in intersections if x.imag < m*x.real+b]
    upper_intersections.sort(key=lambda x: abs(x - B))
    lower_intersections.sort(key=lambda x: abs(x - B))
    if len(lower_intersections) == 0 or len(upper_intersections) == 0:
        plt.plot([test_line.start.real, test_line.end.real],
                 [test_line.start.imag, test_line.end.imag], 'm-')
        return intersections[0], intersections[1]
    return upper_intersections[0], lower_intersections[0]


def make_equidistant(in_vertices, minimum_step):
    # given an input list of vertices, make a new list of vertices that are all
    # minimum_step apart
    def get_diff(i):
        return ((new_vertices[-1][0]-in_vertices[i][0])**2+(new_vertices[-1][1]-in_vertices[i][1])**2)**0.5
    new_vertices = [in_vertices[0]]
    for i in range(1, len(in_vertices)):
        if get_diff(i) < minimum_step:
            continue
        while get_diff(i) >= minimum_step:
            delta_y = in_vertices[i][1]-new_vertices[-1][1]
            delta_x = in_vertices[i][0]-new_vertices[-1][0]
            hyp = (delta_x**2+delta_y**2)**0.5
            new_x = new_vertices[-1][0]+delta_x*minimum_step/hyp
            new_y = new_vertices[-1][1]+delta_y*minimum_step/hyp
            new_vertices.append([new_x, new_y])
    return new_vertices


def initialize_grid(paths):
    current_grid = defaultdict(dict)
    # simplify paths to lines
    poly_paths = []
    for path in paths:
        if path.length() > minimum_stitch:
            num_segments = ceil(path.length() / minimum_stitch)
            for seg_i in range(int(num_segments)):
                poly_paths.append(Line(start=path.point(seg_i/num_segments), end=path.point((seg_i+1)/num_segments)))
        else:
            poly_paths.append(Line(start=path.start, end=path.end))
    bbox = overall_bbox(paths)
    curr_x = int(bbox[0]/minimum_stitch)*minimum_stitch
    total_tests = int(bbox[1]-bbox[0])*int(bbox[3]-bbox[2])/(minimum_stitch*minimum_stitch)
    while curr_x < bbox[1]:
        curr_y = int(bbox[2]/minimum_stitch)*minimum_stitch

        while curr_y < bbox[3]:
            test_line = Line(start=curr_x + curr_y * 1j,
                             end=curr_x + minimum_stitch + (
                                                           curr_y + minimum_stitch) * 1j)
            start = time()
            is_contained = path1_is_contained_in_path2(test_line, Path(*poly_paths))
            end = time()
            print("containment took %.2f, num to look at is %s" % (end-start, total_tests))
            if is_contained:
                current_grid[curr_x][curr_y] = False
            curr_y += minimum_stitch
        curr_x += minimum_stitch
    return current_grid


def scan_lines(paths):
    bbox = overall_bbox(paths)
    lines = []
    fudge_factor = 0.01
    orientation = abs(bbox[3]-bbox[2]) > abs(bbox[1]-bbox[0])
    current_y = bbox[2] if orientation else bbox[0]
    max_pos = bbox[3] if orientation else bbox[1]
    debug_shapes = [[paths, "none", "gray"]]

    while current_y < max_pos:
        current_y += minimum_stitch

        if orientation:
            left = min(bbox[0], bbox[1])
            right = max(bbox[0], bbox[1])

            if left < 0:
                left *= 1.0 + fudge_factor
            else:
                left *= 1.0 - fudge_factor
            if right < 0:
                right *= 1.0 - fudge_factor
            else:
                right *= 1.0 + fudge_factor
            test_line = Line(start=current_y*1j+left, end=current_y*1j+right)
        else:
            up = min(bbox[2], bbox[3])
            down = max(bbox[2], bbox[3])
            if up < 0:
                up *= 1.0 + fudge_factor
            else:
                up *= 1.0 - fudge_factor
            if down < 0:
                down *= 1.0 - fudge_factor
            else:
                down *= 1.0 + fudge_factor
            test_line = Line(start=current_y  + up*1j,
                             end=current_y + down *1j)
        squash_intersections = []
        for path in paths:
            if path.start == path.end:
                continue
            intersections = path.intersect(test_line)
            if len(intersections) > 0:
                squash_intersections += [test_line.point(p[1]) for p in intersections]
        if len(squash_intersections) == 0:
            continue

        intersections = sorted(squash_intersections, key=lambda x: abs(x-test_line.start))
        if len(squash_intersections) < 2:
            continue
        debug_shapes.append([test_line, "none", "black"])
        for i in range(0, 2*int(len(intersections)/2), 2):
            def format_center(ind):
                return (intersections[ind].real, intersections[ind].imag)
            debug_shapes.append([Circle(center=format_center(i), r=1, fill="red")])
            debug_shapes.append([Circle(center=format_center(i+1), r=1, fill="blue")])
            line = Line(start=intersections[i], end=intersections[i+1])
            debug_shapes.append([line, "none", "green"])
            if line.length() > maximum_stitch:
                num_segments = ceil(line.length() / maximum_stitch)
                for seg_i in range(int(num_segments)):
                    lines.append(Line(start=line.point(seg_i/num_segments),
                                      end=line.point((seg_i+1)/num_segments)))
            else:
                lines.append(line)
        write_debug("fillscan", debug_shapes)
    return lines


def fill_test(stitches, scale, current_grid):
    for i in range(1, len(stitches)):
        x_lower = min(stitches[i-1].xx/scale, stitches[i].xx/scale)
        x_upper = max(stitches[i-1].xx/scale, stitches[i].xx/scale)
        y_lower = min(stitches[i-1].yy/scale, stitches[i].yy/scale)
        y_upper = max(stitches[i-1].yy/scale, stitches[i].yy/scale)

        curr_x = int(x_lower/minimum_stitch)*minimum_stitch
        while curr_x <= x_upper+minimum_stitch:
            curr_y = int(y_lower/minimum_stitch)*minimum_stitch
            while curr_y <= y_upper+minimum_stitch:
                if curr_x in current_grid:
                    if curr_y in current_grid[curr_x]:
                        current_grid[curr_x][curr_y] = True
                curr_y += minimum_stitch
            curr_x += minimum_stitch
    last_point = [max(current_grid), 0]
    all_filled = True
    for curr_x in current_grid:
        for curr_y in current_grid[curr_x]:
            if not current_grid[curr_x][curr_y]:
                all_filled = False
                if curr_x < last_point[0] and curr_y > last_point[1]:
                    last_point = [curr_x, curr_y]
    return all_filled, last_point[0]+last_point[1]*1j


def make_continuous(path):
    # takes a discontinuous path (like a donut or a figure 8, and slices it together
    # such that it is continuous
    cont_paths = path.continuous_subpaths()
    paths = cont_paths[0][:]
    for i in range(1, len(cont_paths)):
        start_point = paths[-1].end
        previous_point = paths[0].end
        # find the index of the closest point on the inner circle
        inner_start_index = sorted([(j, abs(path.start - start_point))
                                    for j, path in enumerate(cont_paths)],
                                   key=lambda x: x[1])[0][0]
        next_start_index = sorted([(j, abs(path.start - previous_point))
                                   for j, path in enumerate(cont_paths)
                                   if j != inner_start_index],
                                  key=lambda x: x[1])[0][0]
        paths += [Line(start=start_point, end=cont_paths[i][inner_start_index].start)]
        if next_start_index > inner_start_index:
            paths += cont_paths[i][inner_start_index:] + cont_paths[0:inner_start_index]
        else:
            paths += cont_paths[i][len(
                cont_paths[i]) - inner_start_index:inner_start_index:-1] + cont_paths[
                                                                           inner_start_index::-1]
        paths += [Line(start=cont_paths[-1][inner_start_index].start, end=start_point)]
    return paths


def draw_fill(current_grid, paths):
    colors = ["#dddddd", "#00ff00"]
    draw_paths = [paths]
    for curr_x in current_grid:
        for curr_y in current_grid[curr_x]:
            shape = svgwrite.shapes.Rect(insert=(curr_x, curr_y),
                                         size=(minimum_stitch, minimum_stitch),
                                         fill=colors[current_grid[curr_x][curr_y]])
            draw_paths.insert(0, shape)
    write_debug("draw", paths)



def overall_bbox(paths):
    if not isinstance(paths, list):
        return paths.bbox()
    over_bbox = [None, None, None, None]
    for path in paths:
        bbox = path.bbox()
        for i in range(4):
            if over_bbox[i] is None:
                over_bbox[i] = bbox[i]
        over_bbox[0] = min(over_bbox[0], bbox[0])
        over_bbox[1] = max(over_bbox[1], bbox[1])
        over_bbox[2] = min(over_bbox[2], bbox[2])
        over_bbox[3] = max(over_bbox[3], bbox[3])
    return over_bbox


def remove_close_paths(input_paths):
    if len(input_paths) == 1:
        if input_paths[0].length() < minimum_stitch:
            return []
        else:
            return  input_paths
    def snap_angle(p):
        hyp = p.length()
        y_diff = (p.start-p.end).imag
        if hyp == 0.0:
            return pi*sign(y_diff)/2.0
        elif y_diff/hyp > 1.0:
            return pi/2.0
        elif y_diff/hyp < -1.0:
            return pi/2.0
        else:
            return asin(y_diff/hyp)
    paths = [path for path in input_paths if path.length() >= minimum_stitch]
    # remove any paths that are less than the minimum stitch
    while len([True for line in paths if line.length() < minimum_stitch]) > 0 \
            or len([paths[i] for i in range(1, len(paths)) if
                    paths[i].start != paths[i - 1].end]) > 0:
        paths = [path for path in paths if path.length() >= minimum_stitch]
        paths = [Line(start=paths[i].start, end=paths[(i + 1) % len(paths)].start)
                 for i in range(0, len(paths))]

        angles = [snap_angle(p) for p in paths]
        straight_lines = []
        current_angle = None
        current_start = None
        j = 0
        while j < len(angles):
            if current_angle is None:
                current_angle = angles[0]
                current_start = paths[j].start
            while abs(current_angle - angles[j % len(paths)]) < 0.01:
                j += 1
            straight_lines.append(Line(start=current_start, end=paths[j % len(paths)].start))
            current_angle = angles[j % len(angles)]
            current_start = paths[j % len(paths)].start
        paths = straight_lines
    assert len(
        [i for i in range(1, len(paths)) if paths[i].start != paths[i - 1].end]) == 0
    assert len([True for line in paths if line.length() < minimum_stitch]) == 0
    return paths


def path1_is_contained_in_path2(path1, path2, crosses=False):
    if path2.length() == 0:
        return False
    if path1.start != path1.end:
        if path2.start != path2.end:
            return False
        if path2.intersect(path1):
            return False

    # find a point that's definitely outside path2
    xmin, xmax, ymin, ymax = path2.bbox()
    B = (xmin + 1) + 1j*(ymax + 1)

    A = path1.start  # pick an arbitrary point in path1
    AB_line = Path(Line(A, B))
    number_of_intersections = len(AB_line.intersect(path2))
    if number_of_intersections % 2 and not crosses:  # if number of intersections is odd
        return True
    elif crosses and number_of_intersections > 0:
        return False
    else:
        return False


def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def get_color(v, part="fill"):
    """
    In most svg renderers, if a color is unset, it will be rendered as black. This is
    different from the fill being none, or transparent.
    :param v: the dictionary of attributes from the xml tag
    :param part: the attribute that we're looking for.
    :return: a three item tuple
    """
    if not isinstance(v, dict):
        return [0, 0, 0]
    if "style" not in v:
        if part in v:
            if v[part] in css2_names_to_hex:
                return hex_to_rgb(css2_names_to_hex[v[part]])
            elif v[part][0] == "#":
                return hex_to_rgb(v[part])
            elif v[part] == "none":
                return None
            else:
                return None
        else:
            return None
    if v['style'].find(part + ':') >= 0:
        color = v['style'].split(part + ':')[1].split(";")[0]
        if color[0] == '#':
            return hex_to_rgb(color)
        elif color == "none":
            return None
        else:
            print("not sure what to do with color: %s" % color)
            return None
    else:
        # the color is black
        return None



def sort_paths(paths, attributes):
    # sort paths by colors/ position.
    paths_by_color = defaultdict(list)
    for k, v in enumerate(attributes):
        stroke_color = nearest_color(get_color(v, "stroke"))
        paths_by_color[stroke_color].append(k)
    output_paths = []
    output_attributes = []
    for color in paths_by_color:
        # paths_to_add is a list of indexes of paths in the paths input
        paths_to_add = paths_by_color[color]

        block_bbox = overall_bbox([paths[k] for k in paths_to_add])
        start_location = block_bbox[0] + block_bbox[2] * 1j
        paths_to_add = [(x, 0) for x in paths_to_add] + [(x, 1) for x in paths_to_add]
        while len(paths_to_add) > 0:
            # sort the paths by their distance to the top left corner of the block
            paths_to_add = sorted(paths_to_add,
                                  key=lambda x: abs(
                                      start_location - paths[x[0]].point(x[1])))
            path_to_add = paths_to_add.pop(0)
            output_paths.append(paths[path_to_add[0]] if path_to_add[1]
                                else paths[path_to_add[0]].reversed())
            assert output_paths[-1] is not None
            output_attributes.append(attributes[path_to_add[0]])
            # filter out the reverse path
            paths_to_add = [p for p in paths_to_add if p[0] != path_to_add[0]]
            start_location = paths[path_to_add[0]].start if path_to_add[0] else paths[
                path_to_add[1]].end
            write_debug("sort", [[p, "none", "red"] for p in output_paths] + [[Circle(
                center=(paths[p[0]].point(p[1]).real, paths[p[0]].point(p[1]).imag),
                r=0.1), "gray", "none"] for p in paths_to_add] + [[Circle(
                center=(start_location.real, start_location.imag), r=1), "blue", "none"]])

    # confirm that we haven't abandoned any paths
    assert len(output_paths) == len(paths)
    assert len(output_attributes) == len(attributes)
    assert len(output_attributes) == len(output_paths)
    return output_paths, output_attributes


def shorter_side(paths):
    vector_points = [path.point(0.5) for i, path in enumerate(paths) if i < 4]
    lines = [Line(start=vector_points[0], end=vector_points[2]),
             Line(start=vector_points[1], end=vector_points[3])]
    return 1 if lines[0].length() > lines[1].length() else 0


def gen_filename(partial="anim"):
    return "gen_%s_%s_.svg" % (partial, int(time()*1000))


def write_debug(partial, parts):
    """
    write a set of shapes to an output file.

    :param partial: the filename part, i.e., if partial is xxxx, then the filename will
    be gen_xxxx_timestamp.svg
    :param parts: a list of shapes lists, where the first element of each shape is the
    svgpathtoolshape, the second value is the fill, and the third value is the stroke color.
    :return: nothing
    """
    if not DEBUG:
        return

    debug_fh = open(gen_filename(partial), "w")
    debug_dwg = svgwrite.Drawing(debug_fh, profile='tiny')
    for shape in parts:
        params = {}
        if len(shape) > 2:
            if shape[2] is not None:
                params["stroke"] = rgb_to_hex(shape[2])
        if len(shape) > 1:
            if shape[1] is not None:
                params["fill"] = rgb_to_hex(shape[1])
        if isinstance(shape[0], Path):
            debug_dwg.add(debug_dwg.path(d=shape[0].d(), **params))
        elif isinstance(shape[0], Line):
            debug_dwg.add(debug_dwg.line(start=(shape[0].start.real, shape[0].start.imag),
                                         end=(shape[0].end.real, shape[0].end.imag),
                                         **params))
        elif isinstance(shape[0], svgwrite.shapes.Rect):
            debug_dwg.add(shape[0])
        elif isinstance(shape[0], svgwrite.shapes.Circle):
            debug_dwg.add(shape[0])
        else:
            print("can't put shape", shape[0], " in debug file")
    debug_dwg.write(debug_dwg.filename, pretty=False)
    debug_fh.close()


def stack_paths(all_paths, attributes):
    for i in range(1, len(all_paths)):
        remove_path = all_paths[i].d()
        current_path = all_paths[i-1].d()
        all_paths[i-1] = parse_path(current_path+" "+remove_path)
    return all_paths, attributes


if __name__ == "__main__":
    filename = "cof_orange_hex.svg"
    foldername = dirname(__file__)
    filecontents = open(join(foldername, "workspace", filename), "r").read()
    doc = parseString(filecontents)

    # make sure the document size is appropriate
    root = doc.getElementsByTagName('svg')[0]
    root_width = root.attributes.getNamedItem('width')

    viewbox = root.getAttribute('viewBox')
    all_paths, attributes = svgdoc2paths(doc)
    all_paths, attributes = stack_paths(all_paths, attributes)
    parts = []
    for i in range(len(all_paths)):
        fill_color = get_color(attributes[i], "fill")
        stroke_color = get_color(attributes[i], "stroke")
        parts.append([all_paths[i], fill_color, stroke_color])
    write_debug("stack", parts)
