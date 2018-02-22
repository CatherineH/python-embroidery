from collections import defaultdict
from math import asin
from os import statvfs
from os.path import join, basename, getsize
from shutil import copyfile
from xml.dom.minidom import parseString
from time import time

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

import svgwrite
from numpy import argmax, pi, ceil, sign
from svgpathtools import svgdoc2paths, Line, wsvg, Arc, Path, QuadraticBezier, CubicBezier

from brother import Pattern, Stitch, Block, BrotherEmbroideryFile, pattern_to_csv, \
    pattern_to_svg, nearest_color


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
    #starts_closed = Path(*input_paths).iscontinuous() and input_paths[-1].end == input_paths[0].start
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
    paths = [path for path in input_paths if path.length() > minimum_stitch]
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
    # confirm that we haven't abandoned any paths
    assert len(output_paths) == len(paths)
    assert len(output_attributes) == len(attributes)
    assert len(output_attributes) == len(output_paths)
    return output_paths, output_attributes


# this script is in mms. The assumed minimum stitch width is 1 mm, the maximum width
# is 5 mm
minimum_stitch = 1.0
maximum_stitch = 3.0

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


def cros_stitch_to_pattern(_image):
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
            rgb = "#%02x%02x%02x" % (pixels[color][-1][2][0], pixels[color][-1][2][1], pixels[color][-1][2][2])
            attributes.append({"fill": "none", "stroke": rgb})
    return generate_pattern(output_paths, attributes, 1.0)


def svg_to_pattern(filecontents):
    doc = parseString(filecontents)

    # make sure the document size is appropriate
    root = doc.getElementsByTagName('svg')[0]
    root_width = root.attributes.getNamedItem('width')

    viewbox = root.getAttribute('viewBox')
    all_paths, attributes = sort_paths(*svgdoc2paths(doc))
    print(len(all_paths))

    if root_width is not None:
        root_width = root_width.value
        if root_width.find("mm") > 0:
            root_width = float(root_width.replace("mm", ""))
        elif root_width.find("in") > 0:
            root_width = float(root_width.replace("in", ""))*25.4
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
            if not full_path.iscontinuous():
                print("path not continuous")
                fill_polygon(make_continuous(full_path))
            else:
                fill_polygon(paths)
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
    filename = "dagga_logo.png"
    filecontents = open(join("workspace", filename), "r").read()
    pattern = image_to_pattern(filecontents)
    end = time()
    print("digitizer time: %s" % (end - start))
    pattern_to_csv(pattern, filename + ".csv")
    pattern_to_svg(pattern, filename + ".svg")
    bef = BrotherEmbroideryFile(filename + ".pes")
    bef.write_pattern(pattern)
    upload(filename + ".pes")

    '''
    upload(filename+".pes")
    filename = "text.svg"
    filecontents = open(join("workspace", filename), "r").read()
    pattern = svg_to_pattern(filecontents)
    end = time()
    print("digitizer time: %s" % (end - start))
    pattern_to_csv(pattern, filename + ".csv")
    pattern_to_svg(pattern, filename + ".svg")
    bef = BrotherEmbroideryFile(filename + ".pes")
    bef.write_pattern(pattern)
    '''