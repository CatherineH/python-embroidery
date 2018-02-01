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
from numpy import argmax
from svgpathtools import svgdoc2paths, Line, wsvg, Arc
from svgpathtools.parser import parse_path
from svgpathtools.svg2paths import ellipse2pathd, rect2pathd

from sys import argv
import csv

from brother import Pattern, Stitch, Block, BrotherEmbroideryFile, pattern_to_csv


def overall_bbox(paths):
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
    # assert root.attrib['width'] == '4in'
    # assert root.attrib['height'] == '4in'
    root = doc.getElementsByTagName('svg')[0]
    viewbox = root.getAttribute('viewBox')
    _, attributes = svgdoc2paths(doc)
    # convert all objects to paths:
    for k, v in enumerate(attributes):
        if 'd' not in v:
            if 'cx' in v:
                attributes[k]['d'] = ellipse2pathd(v)
            elif 'x1' in v:
                attributes[k]['d'] = "M %s %s L %s %s" % (
                v['x1'], v['y1'], v['x2'], v['y2'])
            elif 'width' in v and 'height' in v:
                attributes[k]['d'] = rect2pathd(v)
            else:
                print("I'm not sure what to do with %s" % v)
    # The maximum size is 4 inches
    size = 4.0 * 25.4
    if viewbox:
        lims = [float(i) for i in viewbox.split(" ")]
        width = abs(lims[0] - lims[2])
        height = abs(lims[1] - lims[3])
    else:
        # run through all the coordinates
        paths = []
        for k, v in enumerate(attributes):
            paths += parse_path(v['d'])
        bbox = overall_bbox(paths)
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]

    if width > height:
        scale = size / width
    else:
        scale = size / height

    # this script is in mms. The assumed minimum stitch width is 1 mm, the maximum width is 5 mm
    minimum_stitch = 10.0
    maximum_stitch = 15.0

    pattern = Pattern()

    last_color = None
    stitches = []
    dwg = svgwrite.Drawing(stitches_file, profile='tiny')

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
            wsvg(paths, filename=filename)

    def find_intersections(line, v):
        svg_fill = svgwrite.rgb(fill_color[0], fill_color[1], fill_color[2], '%')
        dwg.add(svgwrite.shapes.Line(start=(line.start.real, line.start.imag),
                                     end=(line.end.real, line.end.imag),
                                     stroke="black", stroke_width=minimum_stitch * 0.1))
        paths = parse_path(v['d'])
        intersections = []
        debug_paths = []
        for path in paths:
            if path.start == path.end:
                continue
            debug_paths.append(path)

            dwg.add(dwg.path(v['d'],
                             stroke=svgwrite.rgb(fill_color[0], fill_color[1],
                                                 fill_color[2], '%'), fill="none"))
            #print(line.start == line.end, path.start==path.end)
            line_intersections = line.intersect(path)
            for line_intersection in line_intersections:
                intersection_point = line.point(line_intersection[0])
                epsilon = 1e-4
                if intersection_point in intersections:
                    continue
                intersections.append(intersection_point)
                diffs = intersection_diffs(intersections)
                if len([x for x in diffs if x < epsilon]) > 0:
                    debug_paths.append(line)
                    safe_wsvg(debug_paths, debug)
                    raise ValueError("two intersections are the same!", line, path,
                                     intersections)
        for intersection in intersections:
            dwg.add(dwg.circle(center=(intersection.real, intersection.imag),
                               r=minimum_stitch * 0.1))

        if len(intersections) < 2:
            debug_paths.append(line)
            for path in paths:
                debug_paths.append(path)
            for intersection in intersections:
                debug_paths.append(
                    Line(start=intersection + minimum_stitch + minimum_stitch * 1j,
                         end=intersection - minimum_stitch - minimum_stitch * 1j))
                debug_paths.append(
                    Line(start=intersection - minimum_stitch + minimum_stitch * 1j,
                         end=intersection + minimum_stitch - minimum_stitch * 1j))

            safe_wsvg(debug_paths, debug)
            raise ValueError("only got one intersection!", line)

        dwg.add(svgwrite.shapes.Line(start=(intersections[0].real, intersections[0].imag),
                                     end=(intersections[1].real, intersections[1].imag),
                                     stroke=svg_fill, stroke_width=minimum_stitch * 0.1))
        # grab the point furthest from the start
        lowest = argmax([abs(i - line.start) for i in intersections])

        return intersections[lowest]

    def fill(v):
        fudge_factor = 0.0  # this fudge factor is added to guarantee that the line goes
        # slightly over the object
        paths = parse_path(v['d'])
        bbox = overall_bbox(paths)
        if bbox[0] == bbox[1] or bbox[2] == bbox[3]:
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
                last_stitch = find_intersections(line, v)
            except ValueError as e:
                print("got valueerror on last stitch: %s" % (e))
                break
            to = Stitch(["STITCH"], last_stitch.real *scale, last_stitch.imag * scale,
                        color=fill_color)
            stitches.append(to)
            going_east = not going_east

    for k, v in enumerate(attributes):
        if len(v['d']) > 5000:
            continue
        # first, look for the color from the fill
        fill_color = get_color(v, "fill")
        stroke_color = get_color(v, "stroke")
        print("colors", fill_color, stroke_color)
        if len(pattern.blocks) == 0:
            pattern.add_block(Block([Stitch(["JUMP"], 0, 0)], color=fill_color))

        # first, do the fill - horizontal lines zigzagging from top to bottom
        if last_color != stroke_color:
            stitches = add_block(stitches)
        paths = parse_path(v['d'])
        fill(v)
        last_color = fill_color
        print(pattern)
        add_block(stitches)
        print(pattern)
        # then do the stroke
        if stroke_color is None:
            continue
        if last_color != stroke_color:
            stitches = add_block(stitches)
        for i, path in enumerate(paths):
            if path.length() == 0:
                continue
            if i > 0:
                t = minimum_stitch / path.length()
            else:
                t = 0.0
            if path.length() == 0:
                continue
            # print(type(path))
            while t <= 1.0:
                # print(path.point(t).real * scale, height*scale-path.point(t).imag * scale)
                to = Stitch(["STITCH"], path.point(t).real * scale,
                            path.point(t).imag * scale, color=stroke_color)
                if str(type(path)).find("Line") >= 0:
                    t += 1.0
                    # print(t)
                else:
                    t += minimum_stitch / path.length()
                if len(stitches) == 0:
                    if last_color is not None:
                        block = Block(stitches=[Stitch(["TRIM"], to.xx, to.yy)],
                                      color=last_color)
                        pattern.add_block(block)
                        block = Block(stitches=[Stitch(["COLOR"], to.xx, to.yy)],
                                      color=stroke_color)
                        pattern.add_block(block)

                    print("color", stroke_color)
                    block = Block(stitches=[Stitch(["JUMP"], to.xx, to.yy)],
                                  color=stroke_color)
                    pattern.add_block(block)
                stitches.append(to)
        last_color = stroke_color

    dwg.write(dwg.filename, pretty=False)
    add_block(stitches)
    last_stitch = pattern.blocks[-1].stitches[-1]
    pattern.add_block(
        Block(stitches=[Stitch(["END"], last_stitch.xx, last_stitch.yy)],
              color=pattern.blocks[-1].color))

    print(pattern.thread_count)
    print("thread colors")
    print(pattern.thread_colors)
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
    filename = "discontinuous.svg"#"flowers.svg"
    filecontents = open(join("workspace", filename), "r").read()
    debug_fh = open("debug.svg", "w")
    stitches_fh = open("intersection_test.svg", "w")
    pattern = svg_to_pattern(filecontents, debug_fh, stitches_fh)
    end = time()
    print("digitizer time: %s" % (end - start))
    print("pattern bounds", str(pattern.bounds))
    pattern_to_csv(pattern, filename+".csv")
    bef = BrotherEmbroideryFile(filename + ".pes")
    bef.write_pattern(pattern)
