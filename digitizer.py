from os import statvfs
from os.path import join, basename, getsize
from shutil import copyfile
from xml.dom.minidom import parseString
from time import time

from scipy.spatial.qhull import Voronoi


try:
    # potrace is wrapped in a try/except statement because the digitizer might sometimes
    # be run on an environment where Ctypes are not allowed
    import potrace
    from potrace import BezierSegment, CornerSegment
except:
    potrace = None
    BezierSegment = None
    CornerSegment = None

import matplotlib.pyplot as plt

from numpy import argmax, average, ceil
from svgpathtools import svgdoc2paths, Line, Path

from brother import Pattern, Stitch, Block, BrotherEmbroideryFile, pattern_to_csv, \
    pattern_to_svg, nearest_color
from svgutils import *
from configure import minimum_stitch, maximum_stitch, DEBUG

fill_method = "scan"#"grid"#"polygon"#"voronoi



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
    output_paths, attributes = sort_paths(*stack_paths(*trace_image(filecontents)))
    return generate_pattern(output_paths, attributes, 2.64583333)


def svg_to_pattern(filecontents):
    doc = parseString(filecontents)

    # make sure the document size is appropriate
    root = doc.getElementsByTagName('svg')[0]
    root_width = root.attributes.getNamedItem('width')

    viewbox = root.getAttribute('viewBox')
    all_paths, attributes = sort_paths(*stack_paths(*svgdoc2paths(doc)))

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
    # cut the paths by the paths above
    all_paths, attributes = stack_paths(all_paths, attributes)

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
        vertices = [[average(x[0]), average(x[1])] for x in vertices]
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
        if fill_color is None and stroke_color is None:
            fill_color = [0, 0, 0]
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
    filename = "black_square.png"#"emoji_flag.png"
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