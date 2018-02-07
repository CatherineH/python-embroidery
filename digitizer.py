from os import statvfs, remove
from os.path import join, abspath, basename, getsize, exists
from shutil import copyfile
from xml.dom.minidom import parse, parseString
from time import time
import io

import re
from inspect import getsourcefile
import webcolors

css2_names_to_hex = webcolors.css2_names_to_hex
from webcolors import hex_to_rgb

import svgwrite
from numpy import argmax, pi, cos, sin, ceil
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
    assert path2.isclosed()  # This question isn't well-defined otherwise
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


def svg_to_pattern(filecontents, debug="debug.svg",
                   stitches_file='intersection_test.svg'):
    doc = parseString(filecontents)

    def add_block(stitches):
        if len(stitches) == 0:
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
        print(width, height)
    else:
        # run through all the coordinates
        bbox = overall_bbox(all_paths)
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]

    if width > height:
        scale = size / width
    else:
        scale = size / height

    # this script is in mms. The assumed minimum stitch width is 1 mm, the maximum width
    # is 5 mm
    minimum_stitch = 5.0
    maximum_stitch = 15.0

    pattern = Pattern()

    last_color = None
    stitches = []
    dwg = svgwrite.Drawing(stitches_file, profile='tiny')
    intersection_shapes = []

    def intersection_diffs(intersection):
        diffs = []
        for i in range(len(intersection)):
            for j in range(len(intersection)):
                if i == j:
                    continue
                diffs.append(abs(intersection[i] - intersection[j]))
        return diffs

    def safe_wsvg(paths, filename):
        if not isinstance(filename, str):
            if not filename.closed:
                wsvg(paths, filename=filename)
        else:
            print("filename is a string", type(filename))
            wsvg(paths, filename=filename)

    def fill_test(paths, stitches):
        bbox = overall_bbox(paths)
        curr_x = bbox[0]
        while curr_x < bbox[1]:
            curr_y = bbox[2]
            while curr_y < bbox[3]:
                test_line = Line(start=curr_x+curr_y*1j, end=curr_x+minimum_stitch+(curr_y+minimum_stitch)*1j)
                is_contained = path1_is_contained_in_path2(test_line, Path(*paths))
                if is_contained:
                    # now check to see whether any stitches intersect that bbox
                    
                    shape = svgwrite.shapes.Rect(insert=(curr_x, curr_y), size=(minimum_stitch, minimum_stitch), fill="#dddddd")
                    intersection_shapes.insert(0, shape)
                curr_y += minimum_stitch
            curr_x += minimum_stitch
        print("bbox", bbox)

    def find_intersections(line, v, paths):
        svg_fill = svgwrite.rgb(fill_color[0], fill_color[1], fill_color[2], '%')
        intersection_shapes.append(svgwrite.shapes.Line(start=(line.start.real, line.start.imag),
                                     end=(line.end.real, line.end.imag),
                                     stroke="black", stroke_width=minimum_stitch * 0.1))
        intersections = []
        debug_paths = []
        for path in paths:
            if path.start == path.end:
                continue
            debug_paths.append(path)

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
        intersections = sorted(intersections, key = lambda value: abs(line.start- value))
        # remove any intersections that are within the minimum stitch from each other
        diffs = [abs(intersections[i]-intersections[i-1]) for i in range(1, len(intersections))]
        intersections = [intersections[diff_i] for diff_i in range(len(diffs)) if diffs[diff_i] > minimum_stitch ]+[intersections[-1]]
        if len(intersections) > 2:
            print("sorted intersections:", line.start, intersections)
        elif len(intersections) < 2:
            raise ValueError("too few intersections")

        intersection_shapes.append(svgwrite.shapes.Line(start=(intersections[0].real, intersections[0].imag),
                                     end=(intersections[1].real, intersections[1].imag),
                                     stroke=svg_fill, stroke_width=minimum_stitch * 0.1))

        return intersections[1]

    def fill_zig_zag(v, paths):
        fudge_factor = 0.001  # this fudge factor is added to guarantee that the line goes
        # slightly over the object
        bbox = overall_bbox(paths)
        if bbox[0] == bbox[1] or bbox[2] == bbox[3]:
            return
        # check to see whether the path ends where it starts
        if not paths[0].start == paths[-1].end:
            return
        current_height = bbox[2]
        last_stitch = bbox[0] + 1j * current_height
        # left/right toggle
        going_east = True
        while last_stitch.imag < bbox[3]:
            angle = 0.05
            if going_east:
                line = Line(start=last_stitch,
                            end=bbox[1] * (1.0 + fudge_factor) + 1j * (
                                last_stitch.imag + (bbox[1] - last_stitch.real) * angle))
            else:
                line = Line(end=bbox[0] * (1.0 - fudge_factor) + 1j * (
                    last_stitch.imag + (bbox[1] - last_stitch.real) * angle),
                            start=last_stitch)
            if line.start == line.end:
                raise ValueError(
                    "start is the same as end! last stitch is: %s " % last_stitch)
            try:
                last_stitch = find_intersections(line, v, paths)
            except ValueError as e:
                print("got valueerror on last stitch: %s" % (e))
                break
            if abs(last_stitch-line.start) < minimum_stitch:
                break
            to = Stitch(["STITCH"], last_stitch.real * scale, last_stitch.imag * scale,
                        color=fill_color)
            stitches.append(to)
            going_east = not going_east

    for k, v in enumerate(attributes):
        paths = all_paths[k]
        # first, look for the color from the fill
        fill_color = get_color(v, "fill")
        stroke_color = get_color(v, "stroke")
        if len(pattern.blocks) == 0 and fill_color is not None:
            pattern.add_block(Block([Stitch(["JUMP"], 0, 0)], color=fill_color))

        # first, do the fill - horizontal lines zigzagging from top to bottom
        if last_color != stroke_color:
            stitches = add_block(stitches)
        fill_zig_zag(v, paths)
        fill_test(paths, stitches)
        if last_color is not None and last_color != fill_color and len(stitches) > 0:
            to = stitches[-1]
            block = Block(stitches=[Stitch(["TRIM"], to.xx, to.yy)],
                          color=last_color)
            pattern.add_block(block)

            block = Block(stitches=[Stitch(["COLOR"], to.xx, to.yy)],
                          color=last_color)
            pattern.add_block(block)
        last_color = fill_color
        add_block(stitches)
        # then do the stroke
        if stroke_color is None:
            continue
        if last_color != stroke_color:
            stitches = add_block(stitches)
        for i, path in enumerate(paths):
            if path.length() == 0:
                continue

            to = Stitch(["STITCH"], path.start.real * scale,
                        path.start.imag * scale, color=stroke_color)

            if len(stitches) == 0:
                if "JUMP" not in pattern.blocks[-1].stitches[-1].tags:
                    block = Block(stitches=[Stitch(["TRIM"], to.xx, to.yy)],
                                  color=last_color)
                    pattern.add_block(block)
                    block = Block(stitches=[Stitch(["COLOR"], to.xx, to.yy)],
                                  color=stroke_color)
                    pattern.add_block(block)

                block = Block(stitches=[Stitch(["JUMP"], to.xx, to.yy)],
                              color=stroke_color)
                pattern.add_block(block)
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

        last_color = stroke_color
    for shape in intersection_shapes:
        dwg.add(shape)

    dwg.write(dwg.filename, pretty=False)
    add_block(stitches)
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
    filename = "frame.svg"  # "flowers.svg"
    filecontents = open(join("workspace", filename), "r").read()
    debug_fh = open("debug.svg", "w")
    stitches_fh = open("intersection_test.svg", "w")
    pattern = svg_to_pattern(filecontents, debug_fh, stitches_fh)
    end = time()
    print("digitizer time: %s" % (end - start))
    # print("pattern bounds", str(pattern.bounds))
    pattern_to_csv(pattern, filename + ".csv")
    pattern_to_svg(pattern, filename + ".svg")
    bef = BrotherEmbroideryFile(filename + ".pes")
    bef.write_pattern(pattern)
