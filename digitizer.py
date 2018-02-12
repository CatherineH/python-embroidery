from collections import defaultdict
from os import statvfs, remove
from os.path import join, abspath, basename, getsize, exists
from shutil import copyfile
from xml.dom.minidom import parse, parseString
from time import time
import io

import re
from inspect import getsourcefile
import webcolors
from subprocess import call

css2_names_to_hex = webcolors.css2_names_to_hex
from webcolors import hex_to_rgb

import svgwrite
from numpy import argmax, pi, cos, sin, ceil, average
from svgpathtools import svgdoc2paths, Line, wsvg, Arc, Path, QuadraticBezier
from svgpathtools.parser import parse_path
from svgpathtools.svg2paths import ellipse2pathd, rect2pathd

from sys import argv
import csv

from brother import Pattern, Stitch, Block, BrotherEmbroideryFile, pattern_to_csv, \
    pattern_to_svg


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


def path1_is_contained_in_path2(path1, path2):
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
    if number_of_intersections % 2:  # if number of intersections is odd
        return True
    else:
        return False


def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def get_color(v, part="fill"):
    if "style" not in v:
        if part in v:
            if v[part] in css2_names_to_hex:
                return hex_to_rgb(css2_names_to_hex[v[part]])
            elif v[part][0] == "#":
                return hex_to_rgb(v[part])
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

# this script is in mms. The assumed minimum stitch width is 1 mm, the maximum width
# is 5 mm
minimum_stitch = 3.0
maximum_stitch = 15.0


def initialize_grid(paths):
    current_grid = defaultdict(dict)
    bbox = overall_bbox(paths)
    curr_x = int(bbox[0]/minimum_stitch)*minimum_stitch
    while curr_x < bbox[1]:
        curr_y = int(bbox[2]/minimum_stitch)*minimum_stitch

        while curr_y < bbox[3]:
            test_line = Line(start=curr_x + curr_y * 1j,
                             end=curr_x + minimum_stitch + (
                                                           curr_y + minimum_stitch) * 1j)
            is_contained = path1_is_contained_in_path2(test_line, Path(*paths))
            if is_contained:
                current_grid[curr_x][curr_y] = False
            curr_y += minimum_stitch
        curr_x += minimum_stitch
    return current_grid


def svg_to_pattern(filecontents, debug="debug.svg",
                   stitches_file='intersection_test.svg'):
    doc = parseString(filecontents)
    dwg = svgwrite.Drawing(stitches_file, profile='tiny')
    debug_dwg = svgwrite.Drawing(debug, profile='tiny')

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

    # make sure the document size is appropriate
    root = doc.getElementsByTagName('svg')[0]
    root_width = root.attributes.getNamedItem('width')

    viewbox = root.getAttribute('viewBox')
    all_paths, attributes = svgdoc2paths(doc)
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

    pattern = Pattern()

    last_color = None
    stitches = []
    debug_shapes = []
    intersection_shapes = []

    def intersection_diffs(intersection):
        diffs = []
        for i in range(len(intersection)):
            for j in range(len(intersection)):
                if i == j:
                    continue
                diffs.append(abs(intersection[i] - intersection[j]))
        return diffs

    current_grid = defaultdict(dict)

    def fill_test(paths, stitches):
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

    def draw_fill():
        colors = ["#dddddd", "#00ff00"]
        for curr_x in current_grid:
            for curr_y in current_grid[curr_x]:
                shape = svgwrite.shapes.Rect(insert=(curr_x, curr_y),
                                             size=(minimum_stitch, minimum_stitch),
                                             fill=colors[current_grid[curr_x][curr_y]])
                intersection_shapes.insert(0, shape)

    def find_intersections(line, v, paths):
        svg_fill = svgwrite.rgb(fill_color[0], fill_color[1], fill_color[2], '%')
        intersection_shapes.append(svgwrite.shapes.Line(start=(line.start.real, line.start.imag),
                                     end=(line.end.real, line.end.imag),
                                     stroke="black", stroke_width=minimum_stitch * 0.1))
        intersections = []
        for path in paths:
            if path.start == path.end:
                continue

            intersection_shapes.append(dwg.path(Path(path).d(),
                             stroke=svgwrite.rgb(fill_color[0], fill_color[1],
                                                 fill_color[2], '%'), fill="none"))
            line_intersections = line.intersect(path)
            for line_intersection in line_intersections:
                intersection_point = line.point(line_intersection[0])
                if intersection_point in intersections:
                    continue
                intersections.append(intersection_point)

        for intersection in intersections:
            intersection_shapes.append(dwg.circle(center=(intersection.real, intersection.imag),
                               r=minimum_stitch * 0.1))
        # check to see whether the starting position is in the list of intersections
        diffs = [intersection for intersection in intersections if abs(line.start-intersection) < minimum_stitch]
        if len(diffs) == 0:
            intersections = [line.start]+intersections

        # sort the intersections by their distance to the start point
        intersections = sorted(intersections, key=lambda value: abs(line.start - value))
        # remove any intersections that are within the minimum stitch from each other
        diffs = [abs(intersections[i]-intersections[i-1]) for i in range(1, len(intersections))]
        intersections = [intersections[diff_i] for diff_i in range(len(diffs)) if diffs[diff_i] > minimum_stitch ]+[intersections[-1]]
        if len(intersections) < 2:
            raise ValueError("too few intersections")

        intersection_shapes.append(svgwrite.shapes.Line(start=(intersections[0].real, intersections[0].imag),
                                     end=(intersections[1].real, intersections[1].imag),
                                     stroke=svg_fill, stroke_width=minimum_stitch * 0.1))

        return intersections[1]

    def fill_polygon(paths):

        def remove_close_paths():
            # remove any paths that are less than the minimum stitch
            i = 0
            while i < len(paths):
                if paths[i].length() < minimum_stitch:
                    del paths[i]
                    # confirm that the points merge
                    i = i % len(paths)
                    paths[i].start = paths[i-1].end
                else:
                    i += 1


        def gen_filename():
            return "anim_%s_.svg" % int(time()*1000)

        def fill_trap(paths):
            vector_points = [path.point(0.5) for i, path in enumerate(paths) if i < 4]
            lines = [Line(start=vector_points[0], end=vector_points[2]), Line(start=vector_points[1], end=vector_points[3])]
            side = 1 if lines[0].length() > lines[1].length() else 0
            fill_shape(side, side+2, paths)
            return side

        def fill_shape(side1, side2, paths):
            if paths[side1].length() == 0:
                return
            increment = minimum_stitch/paths[side1].length()
            current_t = 0
            # make closed shape
            filled_paths = [paths[side1], paths[side2]]
            if filled_paths[0].end != filled_paths[1].start:
                filled_paths.insert(1, Line(start=filled_paths[0].end, end=filled_paths[1].start))
            if filled_paths[0].start != filled_paths[-1].end:
                filled_paths.append(Line(start=filled_paths[-1].end,
                                            end=filled_paths[0].start))
            color = "red" if len(filled_paths) == 3 else "green"
            anim_shapes.append(anim_dwg.path(d=Path(*filled_paths).d(), fill=color))
            while current_t < 1.0-increment*0.5:
                point1 = paths[side1].point(current_t)
                point2 = paths[side2].point(1-(current_t+0.5*increment))
                point3 = paths[side1].point(current_t+increment)
                debug_shapes.append(debug_dwg.line(start=(point1.real, point1.imag),
                                                   end=(point2.real, point2.imag), stroke="orange"))
                anim_shapes.append(debug_shapes[-1])
                debug_shapes.append(debug_dwg.line(start=(point2.real, point2.imag),
                                                   end=(point3.real, point3.imag),
                                                   stroke="orange"))
                anim_shapes.append(debug_shapes[-1])
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

        def fill_triangle(paths):
            triangle_sides = [paths[0], paths[1], Line(start=paths[2].start, end=paths[0].start)]
            lengths = [p.length() for p in triangle_sides]
            side1 = argmax(lengths)
            lengths[side1] = 0
            side2 = argmax(lengths)
            fill_shape(side1, side2, triangle_sides)

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
        anim_fh = open(gen_filename(), "w")
        anim_dwg = svgwrite.Drawing(anim_fh, profile='tiny')
        anim_shapes = []
        while len(paths) > 2:
            if len(paths) < 4:
                fill_triangle(paths)
                return
            remove_close_paths()
            # check whether the next triangle is concave
            test_line = Line(start=paths[0].start, end=paths[1].end)
            test_line = Line(start=test_line.point(fudge_factor),
                             end=test_line.point(1 - fudge_factor))
            comparison_path = Path(*paths)
            if not path1_is_contained_in_path2(test_line, Path(*paths)):
                # rotate the paths
                paths = paths[1:]+[paths[0]]
                rotated += 1
                if rotated >= len(paths):
                    print("failed to rotate into a concave path -> ", (test_line.start.real, test_line.start.imag), (test_line.end.real, test_line.end.imag), [(p.start.real, p.start.imag) for p in paths])
                    debug_shapes.append(
                        debug_dwg.line(start=(test_line.start.real, test_line.start.imag),
                                       end=(test_line.end.real, test_line.end.imag),
                                       stroke="blue"))
                    debug_shapes.append(
                        debug_dwg.path(d=comparison_path.d(),
                                       stroke="black", fill="none"))
                    return
                continue

            test_line = Line(start=paths[1].start, end=paths[2].end)
            test_line = Line(start=test_line.point(fudge_factor),
                             end=test_line.point(1 - fudge_factor))
            rect_not_concave = not path1_is_contained_in_path2(test_line, comparison_path)

            # test for concavity. If concave, fill as triangle
            if is_concave(paths) or rect_not_concave:
                fill_triangle(paths)
                to_remove = []
                to_remove.append(paths.pop(0))
                to_remove.append(paths.pop(0))
                for shape in to_remove:
                    anim_shapes.append(anim_dwg.line(start=(shape.start.real, shape.start.imag), end=(shape.end.real, shape.end.imag), fill="none", stroke="blue"))
            else:
                # check whether the next triangle is concave
                side = fill_trap(paths)
                if side:
                    paths = paths[1:]+[paths[0]]
                to_remove = []
                to_remove.append(paths.pop(0))
                to_remove.append(paths.pop(0))
                # if the trap was stitched in the vertical (perpendicular to the
                # stitches), don't remove that segment
                if not side:
                    to_remove.append(paths.pop(0))
                linecolors = ["blue", "purple", "pink"]
                for i, shape in enumerate(to_remove):
                    anim_shapes.append(
                        anim_dwg.line(start=(shape.start.real, shape.start.imag),
                                      end=(shape.end.real, shape.end.imag), fill="none",
                                      stroke=linecolors[i]))
            rotated = 0
            if paths[-1].end != paths[0].start:
                # check for intersections
                closing_line = Line(start=paths[-1].end, end=paths[0].start)
                start_point = paths[-1].end
                intersections = []
                for i, path in enumerate(paths):
                    if path.length() == 0:
                        continue
                    if path.start == closing_line.start and path.end == closing_line.end:
                        continue
                    path_intersections = closing_line.intersect(path)
                    if len(path_intersections) > 0:
                        intersections += [(i, p) for p in path_intersections]

                if len(intersections)> 0:
                    closing_lines = []
                    # I think, but have not proven, that there will always be mod 2
                    # number of intersections
                    for i in range(0, len(intersections)-1, 2):
                        closing_lines.append(Line(start=start_point,
                                                  end=intersections[i][1][0]+intersections[i][1][1]*1j))
                        int_section = intersections[i][0]+1
                        while abs(intersections[i+1][0]-int_section) > 1:
                            if intersections[i+1][0]-int_section > 0:
                                closing_lines.append(paths[int_section])
                                int_section += 1
                            else:
                                closing_lines.append(paths[int_section].reverse())
                                int_section -= 1
                        if intersections[i + 1][0] - int_section > 0:
                            closing_lines.append(Line(start = paths[intersections[i+1][0]].start,
                                                      end=intersections[i+1][1][0]+intersections[i+1][1][1]*1j))
                            start_point = intersections[i+1][1][0]+intersections[i+1][1][1]*1j
                        else:
                            closing_lines.append(
                                Line(start=paths[intersections[i + 1][0]].start,
                                     end=intersections[i + 1][1][0]+intersections[i+1][1][1]*1j))
                            start_point = paths[intersections[i + 1][0]].start
                    closing_lines.append(Line(start=closing_lines[-1].end,
                                              end=paths[-1].start))
                    paths += closing_lines
                else:
                    paths.insert(0, closing_line)

            anim_shapes.append(anim_dwg.path(d=Path(*paths).d(), fill="none", stroke="black"))
            for shape in anim_shapes:
                anim_dwg.add(shape)
            anim_dwg.write(anim_dwg.filename, pretty=False)
            anim_fh.close()
            anim_fh = open(gen_filename(), "w")
            anim_dwg = svgwrite.Drawing(anim_fh, profile='tiny')
            anim_shapes = []

    def make_continuous(path):
        # takes a discontinuous path (like a donut or a figure 8, and slices it together
        # such that it is continuous
        cont_paths = path.continuous_subpaths()
        paths = cont_paths[0][:]
        for i in range(1, len(cont_paths)):
            start_point = paths[-1].end
            paths += [Line(start=start_point, end=cont_paths[i].start)]
            paths += cont_paths[i][:]
            paths += [Line(start=cont_paths[-1].end, end=start_point)]
        full_path = Path(*paths)
        debug_shapes.append(debug_dwg.path(d=full_path.d(), stroke="blue", fill="none"))
        return paths

    def switch_color(stitches, new_color):
        if last_color is not None and last_color != new_color and len(stitches) > 0:
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
        if fill_color == None:
            fill_color = [0, 0, 0]
        stroke_color = get_color(v, "stroke")
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
        add_block(stitches)
        # then do the stroke
        if stroke_color is None:
            continue
        stitches = switch_color(stitches, stroke_color)
        for i, path in enumerate(paths):
            if path.length() == 0:
                continue
            to = Stitch(["STITCH"], path.start.real * scale,
                        path.start.imag * scale, color=stroke_color)
            stitches.append(to)
            # look for intermediary control points
            if isinstance(path, QuadraticBezier):
                # stitch is curved, add an intermediary point
                control_stitch = Stitch(["STITCH"], path.point(0.5).real * scale,
                                        path.point(0.5).imag * scale, color=stroke_color)
                stitches.append(control_stitch)
            elif isinstance(path, Arc):
                # if it's an arc, it's really hard to tell where to put the control
                # points. Instead, let's use the maximum stitch as a guide
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

    for shape in intersection_shapes:
        dwg.add(shape)

    for shape in debug_shapes:
        debug_dwg.add(shape)

    debug_dwg.write(debug_dwg.filename, pretty=False)
    dwg.write(dwg.filename, pretty=False)

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
        statvfs = statvfs(mount_destination)
        if getsize(pes_filename) > statvfs.f_frsize * statvfs.f_bavail:
            print("file cannot be transfered - not enough space on device")
        else:
            copyfile(pes_filename, join(mount_destination, basename(pes_filename)))


if __name__ == "__main__":
    start = time()
    filename = "d.svg"
    filecontents = open(join("workspace", filename), "r").read()
    debug_fh = open("debug_%s_.svg" % time(), "w")
    stitches_fh = open("intersection_test.svg", "w")
    pattern = svg_to_pattern(filecontents, debug_fh, stitches_fh)
    end = time()
    print("digitizer time: %s" % (end - start))
    pattern_to_csv(pattern, filename + ".csv")
    pattern_to_svg(pattern, filename + ".svg")
    bef = BrotherEmbroideryFile(filename + ".pes")
    bef.write_pattern(pattern)
