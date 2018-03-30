from collections import defaultdict
from math import asin
from os import statvfs
from os.path import join, basename, getsize
from shutil import copyfile
from xml.dom.minidom import parseString
from time import time

from scipy.spatial.qhull import Voronoi
from svgwrite.shapes import Circle

try:
    # potrace is wrapped in a try/except statement because the digitizer might sometimes
    # be run on an environment where Ctypes are not allowed
    import potrace
    from potrace import BezierSegment, CornerSegment
except:
    potrace = None
    BezierSegment = None
    CornerSegment = None

import webcolors
from PIL import Image
from StringIO import StringIO
import numpy

css2_names_to_hex = webcolors.css2_names_to_hex
from webcolors import hex_to_rgb
import matplotlib.pyplot as plt

import svgwrite
from numpy import argmax, pi, ceil, sign
from svgpathtools import svgdoc2paths, Line, wsvg, Arc, Path, QuadraticBezier, CubicBezier

from brother import Pattern, Stitch, Block, BrotherEmbroideryFile, pattern_to_csv, \
    pattern_to_svg, nearest_color


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

fill_method = "scan"#"grid"#"polygon"#"voronoi


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
                return [0, 0, 0]
        else:
            return [0, 0, 0]
    if v['style'].find(part + ':') >= 0:
        color = v['style'].split(part + ':')[1].split(";")[0]
        if color[0] == '#':
            return hex_to_rgb(color)
        elif color == "none":
            return None
        else:
            print("not sure what to do with color: %s" % color)
            return [0, 0, 0]
    else:
        # the color is black
        return [0, 0, 0]


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
        paths_to_add = [(x,0) for x in paths_to_add]+[(x, 1) for x in paths_to_add]
        while len(paths_to_add) > 0:
            # sort the paths by their distance to the top left corner of the block
            paths_to_add = sorted(paths_to_add,
                                  key=lambda x: abs(start_location-paths[x[0]].point(x[1])))
            path_to_add = paths_to_add.pop(0)
            output_paths.append(paths[path_to_add[0]] if path_to_add[1]
                                else paths[path_to_add[0]].reversed())
            assert output_paths[-1] is not None
            output_attributes.append(attributes[path_to_add[0]])
            # filter out the reverse path
            paths_to_add = [p for p in paths_to_add if p[0] != path_to_add[0]]
            start_location = paths[path_to_add[0]].start if path_to_add[0] else paths[path_to_add[1]].end
            write_debug("sort", [[p, "none", "red"] for p in output_paths]+[[Circle(center=(paths[p[0]].point(p[1]).real, paths[p[0]].point(p[1]).imag), r=0.1), "gray", "none"] for p in paths_to_add]+[[Circle(center=(start_location.real, start_location.imag), r=1), "blue", "none"]])
            
    # confirm that we haven't abandoned any paths
    assert len(output_paths) == len(paths)
    assert len(output_attributes) == len(attributes)
    assert len(output_attributes) == len(output_paths)
    return output_paths, output_attributes


# this script is in mms. The assumed minimum stitch width is 1 mm, the maximum width
# is 5 mm
minimum_stitch = 3.0
maximum_stitch = 50.0

DEBUG = False


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
        if isinstance(shape[0], Path):
            debug_dwg.add(debug_dwg.path(d=shape[0].d(), fill=shape[1], stroke=shape[2]))
        elif isinstance(shape[0], Line):
            debug_dwg.add(debug_dwg.line(start=(shape[0].start.real, shape[0].start.imag),
                                         end=(shape[0].end.real, shape[0].end.imag),
                                         fill=shape[1], stroke=shape[2]))
        elif isinstance(shape[0], svgwrite.shapes.Rect):
            debug_dwg.add(shape[0])
        elif isinstance(shape[0], svgwrite.shapes.Circle):
            debug_dwg.add(shape[0])
        else:
            print("can't put shape", shape[0], " in debug file")
    debug_dwg.write(debug_dwg.filename, pretty=False)
    debug_fh.close()


def posturize(_image):
    pixels = defaultdict(list)
    for i, pixel in enumerate(_image.getdata()):
        x = i % _image.size[0]
        y = int(i/_image.size[0])
        if len(pixel) > 3:
            if pixel[3] == 255:
                pixels[nearest_color(pixel)].append((x,y, pixel))
        else:
            pixels[nearest_color(pixel)].append((x, y, pixel))
    return pixels


def cross_stitch_to_pattern(_image):
    # this doesn't work well for images with more than 2-3 colors
    max_dimension = max(_image.size)
    pixel_ratio = int(max_dimension*minimum_stitch/(4*25.4))
    if pixel_ratio != 0:
        _image = _image.resize((_image.size[0]/pixel_ratio, _image.size[1]/pixel_ratio))
    pixels = posturize(_image)

    paths = []
    attrs = []

    for color in pixels:
        for pixel in pixels[color]:
            rgb = "#%02x%02x%02x" % (pixel[2][0], pixel[2][1], pixel[2][2])
            x = pixel[0]
            y = pixel[1]
            attrs.append({"fill": "none", "stroke": rgb})
            paths.append(Path(Line(start=x + 1j * y,
                                   end=x + 0.5 * minimum_stitch + 1j * (y + minimum_stitch))))
    debug_paths = [[path, attrs[i]["fill"], attrs[i]["stroke"]] for i, path in enumerate(paths)]
    write_debug("png", debug_paths)
    return generate_pattern(paths, attrs, 1.0)


def image_to_pattern(filecontents):
    output = StringIO()
    output.write(filecontents)
    _image = Image.open(output)
    pixels = posturize(_image)
    output_paths = []
    attributes = []
    for color in pixels:
        data = numpy.zeros(_image.size, numpy.uint32)
        for pixel in pixels[color]:
            data[pixel[0], pixel[1]] = 1
        # Create a bitmap from the array
        bmp = potrace.Bitmap(data)
        # Trace the bitmap to a path
        path = bmp.trace()
        # Iterate over path curves
        for curve in path:
            svg_paths = []
            start_point = curve.start_point
            for segment in curve:
                if isinstance(segment, BezierSegment):
                    svg_paths.append(CubicBezier(start=start_point[1]+1j*start_point[0],
                                                 control1=segment.c1[1]+segment.c1[0]*1j,
                                                 control2=segment.c2[1] + segment.c2[0] * 1j,
                                                 end=segment.end_point[1]+1j*segment.end_point[0]))
                elif isinstance(segment, CornerSegment):
                    svg_paths.append(Line(start=start_point[1] + 1j * start_point[0],
                                    end=segment.c[1] + segment.c[0] * 1j))
                    svg_paths.append(Line(start=segment.c[1] + segment.c[0] * 1j,
                                    end=segment.end_point[1] + 1j * segment.end_point[0]))
                else:
                    print("not sure what to do with: ", segment)
                start_point = segment.end_point
            output_paths.append(Path(*svg_paths))
            color = pixel[2]
            rgb = "#%02x%02x%02x" % (color[0], color[1], color[2])
            # is the path closed?
            fill = rgb if output_paths[0].start == output_paths[-1].end else "none"
            attributes.append({"fill": fill, "stroke": rgb})
    return generate_pattern(output_paths, attributes, 1.0)





def svg_to_pattern(filecontents):
    doc = parseString(filecontents)

    # make sure the document size is appropriate
    root = doc.getElementsByTagName('svg')[0]
    root_width = root.attributes.getNamedItem('width')

    viewbox = root.getAttribute('viewBox')
    all_paths, attributes = sort_paths(*svgdoc2paths(doc))

    if root_width is not None:
        root_width = root_width.value
        if root_width.find("mm") > 0:
            root_width = float(root_width.replace("mm", ""))
        elif root_width.find("in") > 0:
            root_width = float(root_width.replace("in", ""))*25.4
        elif root_width.find("px") > 0:
            root_width = float(root_width.replace("px", ""))*0.264583333
        else:
            root_width = float(root_width)
    # The maximum size is 4 inches - multiplied by 10 for scaling
    if root_width:
        size = root_width
    size *= 10.0
    if viewbox:
        lims = [float(i) for i in viewbox.split(" ")]
        width = abs(lims[0] - lims[2])
        height = abs(lims[1] - lims[3])
    else:
        # run through all the coordinates
        bbox = overall_bbox(all_paths)
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]

    if width > height:
        scale = size / width
    else:
        scale = size / height
    return generate_pattern(all_paths, attributes, scale)


def generate_pattern(all_paths, attributes, scale):

    def add_block(stitches):
        if len(stitches) == 0:
            print("got no stitches in add block!")
            return []
        if last_color is not None:
            block = Block(stitches=stitches, color=last_color)
            pattern.add_block(block)

        else:
            print("last color was none, not adding the block")
        return []
    pattern = Pattern()

    last_color = None
    stitches = []

    def fill_polygon(paths):
        def fill_trap(paths,color="gray"):
            side = shorter_side(paths)
            shapes = [[Path(*paths), "none", "black"], [Path(*paths[side:side+3]), color, "none"]]
            side2 = side+2
            shapes = fill_shape(side, side2, paths, shapes)
            write_debug("fill", shapes)
            return side, side2

        def fill_shape(side1, side2, paths, shapes):
            if paths[side1].length() == 0:
                return
            increment = 3*minimum_stitch/paths[side1].length()
            current_t = 0
            # make closed shape

            filled_paths = [paths[side1], paths[side2]]
            if filled_paths[0].end != filled_paths[1].start:
                filled_paths.insert(1, Line(start=filled_paths[0].end, end=filled_paths[1].start))
            if filled_paths[0].start != filled_paths[-1].end:
                filled_paths.append(Line(start=filled_paths[-1].end,
                                            end=filled_paths[0].start))
            while current_t < 1.0-increment*0.5:
                point1 = paths[side1].point(current_t)
                point2 = paths[side2].point(1-(current_t+0.5*increment))
                point3 = paths[side1].point(current_t+increment)
                to = Stitch(["STITCH"], point1.real * scale, point1.imag * scale,
                            color=fill_color)
                stitches.append(to)
                to = Stitch(["STITCH"], point2.real * scale, point2.imag * scale,
                            color=fill_color)
                stitches.append(to)
                current_t += increment
                to = Stitch(["STITCH"], point3.real * scale, point3.imag * scale,
                            color=fill_color)
                stitches.append(to)
            shapes.append([paths[side1], "none", "orange"])
            shapes.append([paths[side2], "none", "red"])
            return shapes

        def fill_triangle(paths, color="green"):
            triangle_sides = [paths[0], paths[1], Line(start=paths[2].start, end=paths[0].start)]
            shapes = [[Path(*paths), "none", "black"], [Path(*triangle_sides), color, "none"]]
            lengths = [p.length() for p in triangle_sides]
            side1 = argmax(lengths)
            lengths[side1] = 0
            side2 = argmax(lengths)
            shapes = fill_shape(side1, side2, triangle_sides, shapes)
            write_debug("fill", shapes)

        def is_concave(paths):
            xs = [path.start.real for i, path in enumerate(paths) if i < 4]
            ys = [path.start.imag for i, path in enumerate(paths) if i < 4]
            x_range = [min(xs), max(xs)]
            y_range = [min(ys), max(ys)]
            for i in range(0, 4):
                p = paths[i].start
                if x_range[0] < p.real < x_range[1] and y_range[0] < p.imag < y_range[1]:
                    return True
            return False
        rotated = 0
        fudge_factor = 0.03
        while len(paths) > 2:
            if len(paths) < 4:
                fill_triangle(paths, color="red")
                return
            shapes = [[Path(*paths), "none", "blue"], [Path(*paths), "none", "green"]]
            write_debug("close", shapes)
            paths = remove_close_paths(paths)

            if len(paths) <= 2:
                return
            # check whether the next triangle is concave
            test_line1 = Line(start=paths[0].start, end=paths[1].end)
            test_line1 = Line(start=test_line1.point(fudge_factor),
                             end=test_line1.point(1 - fudge_factor))
            comparison_path = Path(*paths)
            if test_line1.length() == 0:
                has_intersection = True
            else:
                has_intersection = len([1 for line in paths if len(line.intersect(test_line1)) > 0]) > 0

            if not path1_is_contained_in_path2(test_line1, comparison_path) or has_intersection:
                shapes = [[comparison_path, "none", "blue"], [test_line1, "none", "black"]]
                write_debug("anim", shapes)
                # rotate the paths
                paths = paths[1:]+[paths[0]]
                rotated += 1
                if rotated >= len(paths):
                    print("failed to rotate into a concave path -> ", (test_line1.start.real, test_line1.start.imag), (test_line1.end.real, test_line1.end.imag), [(p.start.real, p.start.imag) for p in paths])
                    return
                continue
            side = shorter_side(paths)

            test_line2 = Line(start=paths[1].start, end=paths[2].end)
            test_line2 = Line(start=test_line2.point(fudge_factor),
                             end=test_line2.point(1 - fudge_factor))
            test_line3 = Line(start=paths[-1+side].end, end=paths[(3+side) % len(paths)].start)
            test_line3 = Line(start=test_line3.point(fudge_factor),
                              end=test_line3.point(1 - fudge_factor))

            num_intersections = []
            for path in comparison_path:
                if test_line3.length() == 0:
                    print("test line 3 is degenerate!")
                num_intersections += test_line3.intersect(path)
                num_intersections += test_line2.intersect(path)

            rect_not_concave = not path1_is_contained_in_path2(test_line2, comparison_path)

            # test for concavity. If concave, fill as triangle
            if is_concave(paths) or len(num_intersections) > 0 or rect_not_concave:
                fill_triangle(paths, color="blue")
                shapes = [[Path(*paths), "none", "black"]]
                to_remove = []
                to_remove.append(paths.pop(0))
                to_remove.append(paths.pop(0))
                for shape in to_remove:
                    shapes.append([shape, "none", "blue"])
                closing_line = Line(start=paths[-1].end, end=paths[0].start)
                shapes.append([closing_line, "none", "green"])
                shapes.append([test_line1, "none", "red"])
                write_debug("rem", shapes)

            else:
                # check whether the next triangle is concave
                side, side2 = fill_trap(paths)
                if side:
                    paths = paths[1:]+[paths[0]]
                shapes = [[Path(*paths), "none", "black"]]
                to_remove = []
                to_remove.append(paths.pop(0))
                to_remove.append(paths.pop(0))
                to_remove.append(paths.pop(0))
                # if the trap was stitched in the vertical (perpendicular to the
                # stitches), don't remove that segment
                linecolors = ["blue", "purple", "pink"]
                for i,shape in enumerate(to_remove):
                    shapes.append([shape, "none", linecolors[i]])
                closing_line = Line(start=paths[-1].end, end=paths[0].start)
                shapes.append([closing_line, "none", "green"])
                shapes.append([test_line2, "none", "purple"])
                write_debug("rem", shapes)
                delta = closing_line.length()-(test_line3.length()/(1.0-2.0*fudge_factor))
                if abs(delta) > 1e-14:
                    print("closing line different than test!", side, test_line3, closing_line)
            rotated = 0
            if paths[-1].end != paths[0].start:
                # check for intersections
                closing_line = Line(start=paths[-1].end, end=paths[0].start)
                paths.insert(0, closing_line)
            else:
                print("removed paths but they connected anyway")

    def switch_color(stitches, new_color):
        if last_color is not None and last_color != new_color and len(stitches) > 0:
            add_block(stitches)
            to = stitches[-1]
            block = Block(stitches=[Stitch(["TRIM"], to.xx, to.yy)],
                          color=last_color)
            pattern.add_block(block)
            block = Block(stitches=[Stitch(["COLOR"], to.xx, to.yy)],
                          color=new_color)
            pattern.add_block(block)
            return []
        return stitches

    def fill_grid(paths):
        grid = initialize_grid(paths)

        def grid_available(pos):
            if pos.real in grid:
                if pos.imag in grid[pos.real]:
                    return not grid[pos.real][pos.imag]
                else:
                    return False
            else:
                return False

        def count_empty():
            count = 0
            for x in grid:
                for y in grid[x]:
                    count += not grid[x][y]
            return count

        draw_fill(grid, paths)
        # need to find the next location to stitch to. It needs to zig-zag, so we need to
        # keep a record of what direction it was going in
        going_east = True

        def find_upper_corner():
            # find the top or bottom left corner of the grid
            curr_pos = None
            for x in grid:
                for y in grid[x]:
                    if curr_pos is None:
                        curr_pos = x + 1j*y
                        continue
                    if x < curr_pos.real and y < curr_pos.imag and not grid[x][y]:
                        curr_pos = x + 1j*y
            return curr_pos

        rounds = 1
        num_empty = count_empty()
        while num_empty > 0:
            print("round %s, still empty %s" % (rounds, num_empty))
            curr_pos = find_upper_corner()
            to = Stitch(["STITCH"], curr_pos.real * scale, curr_pos.imag * scale,
                        color=fill_color)
            stitches.append(to)
            blocks_covered = int(maximum_stitch/minimum_stitch)
            while grid_available(curr_pos):
                for i in range(0, blocks_covered):
                    sign = 1.0 if going_east else -1.0
                    test_pos = curr_pos+sign*i*minimum_stitch
                    if not grid_available(test_pos):
                        break
                    else:
                        next_pos = test_pos + 1j*minimum_stitch
                going_east = not going_east
                to = Stitch(["STITCH"], next_pos.real * scale, next_pos.imag * scale,
                            color=fill_color)
                stitches.append(to)
                curr_pos = next_pos
            draw_fill(grid, paths)
            new_num_empty = count_empty()
            if new_num_empty == num_empty:
                print("fill was not able to fill any parts of the grid!")
                break
            else:
                num_empty = new_num_empty
            rounds += 1

    def fill_scan(paths):
        lines = scan_lines(paths)
        attributes = [{"stroke": fill_color} for i in range(len(lines))]
        print("before lines %s " % len(lines))
        lines, attributes = sort_paths(lines, attributes)
        print("after lines %s " % len(lines))
        if isinstance(lines, list):
            if len(lines) == 0:
                return
            start_point = lines[0].start
        else:
            start_point = lines.start
        to = Stitch(["STITCH"], start_point.real * scale,
                    start_point.imag * scale, color=fill_color)
        stitches.append(to)

        for line in lines:

            to = Stitch(["STITCH"], line.start.real * scale,
                        line.start.imag * scale, color=fill_color)
            stitches.append(to)
            to = Stitch(["STITCH"], line.end.real * scale,
                        line.end.imag * scale, color=fill_color)
            stitches.append(to)
        print("stitch length", len(stitches))

    def fill_voronoi(paths):
        points = []
        for path in paths:
            num_stitches = 100.0*path.length()/maximum_stitch
            ppoints = [path.point(i/num_stitches) for i in range(int(num_stitches))]
            for ppoint in ppoints:
                points.append([ppoint.real, ppoint.imag])
            points.append([path.end.real, path.end.imag])
        vor = Voronoi(points)
        vertices = vor.vertices

        pxs = [x[0] for x in points]
        pys = [-x[1] for x in points]
        plt.plot(pxs, pys)
        # restrict the points to ones within the shape
        vertices = [x for i, x in enumerate(vertices)
                    if path1_is_contained_in_path2(Line(end=x[0]+x[1]*1j,
                                                        start=x[0]+0.01+x[1]*1j),
                                                   Path(*paths))]
        # now sort the vertices. This is close but not quite what is being done in
        # sort_paths
        new_vertices = []
        start_location = points[0]
        while len(vertices) > 0:
            vertices = sorted(vertices,
                              key=lambda x: (start_location[0] - x[0])**2
                                            +(start_location[1]-x[1])**2)
            new_vertices.append(vertices.pop(0))
            start_location = new_vertices[-1]
        vertices = new_vertices
        # now smooth out the vertices
        vertices = [[[x[0] for x in vertices[i:i + 3]],
                     [x[1] for x in vertices[i:i + 3]]]
                    for i in range(0, len(vertices) - 3)]
        vertices = [[numpy.average(x[0]), numpy.average(x[1])] for x in vertices]
        # we want each vertice to be about equidistant
        vertices = make_equidistant(vertices, minimum_stitch/2.0)
        xs = [x[0] for x in vertices]
        ys = [-x[1] for x in vertices]

        plt.plot(xs, ys, 'r-')
        stitchx = [vertices[0][0]]
        stitchy = [vertices[0][1]]

        # make spines
        for i in range(len(vertices) - 1):
            intersections = perpendicular(vertices[i][0]+vertices[i][1]*1j,
                                          vertices[i + 1][0]+vertices[i+1][1]*1j,
                                          Path(*paths))
            diff = abs(intersections[0]-intersections[1])
            if diff > 9:
                continue
            stitchx.append(intersections[0].real)
            stitchy.append(-intersections[0].imag)
            stitchx.append(intersections[1].real)
            stitchy.append(-intersections[1].imag)
        for i in range(len(stitchx)):
            to = Stitch(["STITCH"], stitchx[i] * scale,
                        -stitchy[i] * scale, color=fill_color)
            stitches.append(to)

        plt.plot(stitchx, stitchy, 'g-')
        plt.xlim(min(pxs), max(pxs))
        plt.ylim(min(pys), max(pys))
        #plt.show()

    for k, v in enumerate(attributes):
        paths = all_paths[k]
        # first, look for the color from the fill
        fill_color = get_color(v, "fill")
        stroke_color = get_color(v, "stroke")
        # if both the fill color and stroke color are none,
        if fill_color is not None:
            if len(pattern.blocks) == 0 and fill_color is not None:
                pattern.add_block(Block([Stitch(["JUMP"], 0, 0)], color=fill_color))
            stitches = switch_color(stitches, fill_color)
            full_path = Path(*paths)
            if fill_method == "polygon":
                if not full_path.iscontinuous():
                    print("path not continuous")
                    fill_polygon(make_continuous(full_path))
                else:
                    fill_polygon(paths)
            elif fill_method == "grid":
                fill_grid(paths)
            elif fill_method == "scan":
                fill_scan(paths)
            elif fill_method == "voronoi":
                fill_voronoi(paths)
            last_color = fill_color
        if fill_color is not None:
            add_block(stitches)
        # then do the stroke
        if stroke_color is None:
            continue
        stitches = switch_color(stitches, stroke_color)
        #paths = remove_close_paths(paths)
        for i, path in enumerate(paths):
            if path.length() == 0:
                continue
            to = Stitch(["STITCH"], path.start.real * scale,
                        path.start.imag * scale, color=stroke_color)
            stitches.append(to)
            num_segments = ceil(path.length()/maximum_stitch)
            for seg_i in range(int(num_segments+1)):
                control_stitch = Stitch(["STITCH"], path.point(seg_i/num_segments).real * scale,
                                    path.point(seg_i/num_segments).imag * scale, color=stroke_color)
                stitches.append(control_stitch)

            # if the next stitch doesn't start at the end of this stitch, add that one as
            # well
            end_stitch = Stitch(["STITCH"], path.end.real * scale,
                                path.end.imag * scale, color=stroke_color)
            if i != len(paths) - 1:
                if path.end != paths[i + 1].start:
                    stitches.append(end_stitch)
            else:
                stitches.append(end_stitch)
        if len(stitches) > 0:
            last_color = stroke_color
    add_block(stitches)
    if len(pattern.blocks[-1].stitches) > 0:
        last_stitch = pattern.blocks[-1].stitches[-1]
        pattern.add_block(
            Block(stitches=[Stitch(["END"], last_stitch.xx, last_stitch.yy)],
                color=pattern.blocks[-1].color))

    return pattern


def upload(pes_filename):
    # since this library is sometimes used in web-environments, where subprocesses aren't
    # allowed, move this import statement to only where it's actually used.
    from subprocess import check_output
    mount_points = {}
    out = check_output(["df"]).split("\n")
    for line in out:
        if line.find("/media") >= 0:
            mount_location = line.split("%")[-1]
            line = line.split(" ")
            mount_points[line[0]] = mount_location.strip()

    mount_destination = None
    for mount_point in mount_points:
        out = check_output(
            ["udevadm", "info", "--name=" + mount_point, "--attribute-walk"])
        if out.find('ATTRS{idVendor}=="04f9"') > 0:
            print("found brother device, assuming embroidery machine")
            mount_destination = mount_points[mount_point]
            break

    if mount_destination is None:
        print("could not find brother embroidery machine to transfer to")
    else:
        _statvfs = statvfs(mount_destination)
        if getsize(pes_filename) > _statvfs.f_frsize * _statvfs.f_bavail:
            print("file cannot be transfered - not enough space on device")
        else:
            copyfile(pes_filename, join(mount_destination, basename(pes_filename)))


if __name__ == "__main__":
    start = time()
    filename = "cof_orange_hex.svg"
    filecontents = open(join("workspace", filename), "r").read()
    if filename.split(".")[-1] != "svg":
        pattern = image_to_pattern(filecontents)
    else:
        pattern = svg_to_pattern(filecontents)
    end = time()
    print("digitizer time: %s" % (end - start))
    pattern_to_csv(pattern, filename + ".csv")
    pattern_to_svg(pattern, filename + ".svg")
    bef = BrotherEmbroideryFile(filename + ".pes")
    bef.write_pattern(pattern)
    upload(filename + ".pes")