from os import statvfs, remove
from os.path import join, abspath, basename, getsize, exists
from shutil import copyfile
from xml.dom.minidom import parse, parseString
from time import time
import io

import re

import svgwrite
from numpy import argmax
from svgpathtools import svgdoc2paths, Line, wsvg, Arc
from sys import argv
import csv

from brother import Pattern, Stitch, Block, BrotherEmbroideryFile


def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def get_color(v, part="fill"):
    if v['style'].find(part+':') >= 0:
        color = v['style'].split(part+':')[1].split(";")[0]
        if color[0] == '#':
            return [int(color[1:3], 16), int(color[3:5], 16), int(color[5:], 16)]
        elif color == "none":
            return None
        else:
            print("not sure what to do with color: %s" % color)
            return [0, 0, 0]
    else:
        # the color is black
        return [0, 0, 0]


def svg_to_pattern(filecontents, debug="debug.svg", stitches_file='intersection_test.svg'):
    doc = parseString(filecontents)

    def add_block(stitches):
        if last_color is not None:
            block = Block(stitches=stitches, color=last_color)
            pattern.blocks.append(block)
        return []
    # make sure the document size is appropriate
    #assert root.attrib['width'] == '4in'
    #assert root.attrib['height'] == '4in'
    root = doc.getElementsByTagName('svg')[0]
    viewbox = root.getAttribute('viewBox')
    paths, attributes = svgdoc2paths(doc)
    # The maximum size is 4 inches
    size = 4.0 * 25.4
    if viewbox:
        lims = [float(i) for i in viewbox.split(" ")]
        width = abs(lims[0] - lims[2])
        height = abs(lims[1] - lims[3])
    else:
        # run through all the coordinates
        xs = []
        ys = []
        for k, v in enumerate(attributes):
            for path in paths[k]:
                bbox = path.bbox()
                xs.append(bbox[0])
                xs.append(bbox[1])
                ys.append(bbox[2])
                ys.append(bbox[3])
        width = max(xs)-min(xs)
        height = max(ys)-min(ys)

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
                diffs.append(abs(intersection[i]-intersection[j]))
        return diffs

    def find_intersections(line, v, k, paths):
        svg_fill = svgwrite.rgb(fill_color[0], fill_color[1], fill_color[2], '%')
        dwg.add(svgwrite.shapes.Line(start=(line.start.real, line.start.imag),
                                     end=(line.end.real, line.end.imag),
                                     stroke="black", stroke_width=minimum_stitch * 0.1))
        intersections = []
        debug_paths = []
        for path in paths[k]:
            debug_paths.append(path)
            dwg.add(dwg.path(v['d'],
                             stroke=svgwrite.rgb(fill_color[0], fill_color[1],
                                                 fill_color[2], '%'), fill="none"))
            line_intersections = line.intersect(path)
            for line_intersection in line_intersections:
                intersections.append(line.point(line_intersection[0]))
                epsilon = 1e-4
                diffs = intersection_diffs(intersections)
                if len([x for x in diffs if x < epsilon]) > 0:
                    debug_paths.append(line)
                    wsvg(debug_paths, filename=debug)
                    raise ValueError("two intersections are the same!", line, path)
        for intersection in intersections:
            dwg.add(dwg.circle(center=(intersection.real, intersection.imag),
                               r=minimum_stitch * 0.1))

        if len(intersections) < 2:
            debug_paths.append(line)
            for path in paths[k]:
                debug_paths.append(path)
            for intersection in intersections:
                debug_paths.append(Line(start=intersection+minimum_stitch+minimum_stitch*1j,
                                        end=intersection-minimum_stitch-minimum_stitch*1j))
                debug_paths.append(
                    Line(start=intersection - minimum_stitch + minimum_stitch*1j,
                         end=intersection + minimum_stitch - minimum_stitch*1j))
            wsvg(debug_paths, filename=debug)
            raise ValueError("only got one intersection!", line)

        dwg.add(svgwrite.shapes.Line(start=(intersections[0].real, intersections[0].imag),
                                     end=(intersections[1].real, intersections[1].imag),
                                     stroke=svg_fill, stroke_width=minimum_stitch * 0.1))
        lowest = argmax([i.imag for i in intersections])

        dwg.save()
        return intersections[lowest]

    for k, v in enumerate(attributes):
        # if k > 0:
        #    continue
        # first, look for the color from the fill
        fill_color = get_color(v, "fill")
        stroke_color = get_color(v, "stroke")

        if len(pattern.blocks) == 0:
            pattern.blocks.append(Block([Stitch(["JUMP"], 0, 0)], color=fill_color))

        # first, do the fill - horizontal lines zigzagging from top to bottom
        if last_color != stroke_color:
            stitches = add_block(stitches)
        current_height = paths[k].bbox()[2]
        last_stitch = paths[k].bbox()[0]+ 1j*current_height
        while last_stitch.imag < paths[k].bbox()[3]:
            angle = 0.05
            line = Line(start=last_stitch,
                        end=paths[k].bbox()[1] + 1j*(last_stitch.imag+(paths[k].bbox()[1]-last_stitch.real)*angle))
            try:
                east_stitch = find_intersections(line, v, k, paths)
            except ValueError:
                break
            line = Line(end=paths[k].bbox()[0] + 1j * (east_stitch.imag +(paths[k].bbox()[1]-east_stitch.real)*angle),
                        start=east_stitch)
            try:
                last_stitch = find_intersections(line, v, k, paths)
            except ValueError:
                break

        # then do the stroke
        if stroke_color is None:
            continue
        if last_color != stroke_color:
            stitches = add_block(stitches)
        for i, path in enumerate(paths[k]):
            if path.length() == 0:
                continue
            if i > 0:
                t = minimum_stitch / path.length()
            else:
                t = 0.0
            if path.length() == 0:
                continue
            #print(type(path))
            while t <= 1.0:
                #print(path.point(t).real * scale, height*scale-path.point(t).imag * scale)
                to = Stitch(["STITCH"], path.point(t).real * scale, path.point(t).imag * scale, color=stroke_color)
                if str(type(path)).find("Line") >= 0:
                    t += 1.0
                    #print(t)
                else:
                    t += minimum_stitch / path.length()
                if len(stitches) == 0:
                    if last_color is not None:
                        block = Block(stitches=[Stitch(["TRIM"], to.xx, to.yy)],
                                      color=last_color)
                        pattern.blocks.append(block)
                        block = Block(stitches=[Stitch(["COLOR"], to.xx, to.yy)],
                                      color=stroke_color)
                        pattern.blocks.append(block)

                    print("color", stroke_color)
                    block = Block(stitches=[Stitch(["JUMP"], to.xx, to.yy)], color=stroke_color)
                    pattern.blocks.append(block)
                stitches.append(to)
        last_color = stroke_color
    add_block(stitches)
    last_stitch = pattern.blocks[-1].stitches[-1]
    pattern.blocks.append(Block(stitches=[Stitch(["END"], last_stitch.xx, last_stitch.yy)],
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
        out = check_output(["udevadm", "info", "--name="+mount_point, "--attribute-walk"])
        if out.find('ATTRS{idVendor}=="04f9"')> 0:
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
    filename = "circles.svg"#"ns_flag.svg"#"flowers.svg"
    filecontents = open(join("workspace", filename), "r").read()
    debug_fh = open("debug.svg", "w")
    stitches_fh = open("intersection_test.svg", "w")
    pattern = svg_to_pattern(filecontents, debug_fh, stitches_fh)
    end = time()
    print("digitizer time: %s" % (end-start))
    print("pattern bounds", str(pattern.bounds))
    bef = BrotherEmbroideryFile(filename + ".pes")
    bef.write_pattern(pattern)
