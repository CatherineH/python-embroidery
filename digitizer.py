import argparse

from block import Block
from pattern import Pattern
from pattern_utils import de_densify, measure_density, pattern_to_svg
from stitch import Stitch
from svgutils import *


if PLOTTING:
    from scipy.spatial.qhull import Voronoi
    import matplotlib.pyplot as plt
else:
    plt = None

try:
    # potrace is wrapped in a try/except statement because the digitizer might sometimes
    # be run on an environment where Ctypes are not allowed
    import potrace
    from potrace import BezierSegment, CornerSegment
except:
    potrace = None
    BezierSegment = None
    CornerSegment = None

from numpy import argmax, average, ceil
from svgpathtools import svgdoc2paths, Line, Path

from brother import BrotherEmbroideryFile, pattern_to_csv, upload
from configure import minimum_stitch, maximum_stitch, DEBUG

fill_method = "scan" #"grid"#"polygon"#"voronoi

parser = argparse.ArgumentParser(
    description='Generate a pes file for brother sewing machines from an svg or png image')
parser.add_argument('--filename', type=str,
                    help='The filename of the input image.')
parser.add_argument('--fill', dest="fill", action="store_true",
                    help="Fill the shapes")


class Digitizer(object):
    def __init__(self, filename=None, fill=False):
        self.fill = fill
        # stitches is the stitches that have yet to be added to the pattern
        self.stitches = []
        self.attributes = []
        self.all_paths = []
        self.fill_color = None
        self.last_color = None
        self.pattern = Pattern()

        if not filename:
            return
        self.filecontents = open(join("workspace", filename), "r").read()
        if filename.split(".")[-1] != "svg":
            self.image_to_pattern()
        else:
            self.svg_to_pattern()

    def image_to_pattern(self):
        self.all_paths, self.attributes = stack_paths(*trace_image(self.filecontents))
        self.scale = 2.64583333
        self.generate_pattern()

    def svg_to_pattern(self):
        doc = parseString(self.filecontents)

        # make sure the document size is appropriate
        root = doc.getElementsByTagName('svg')[0]
        root_width = root.attributes.getNamedItem('width')

        viewbox = root.getAttribute('viewBox')
        if self.fill:
            self.all_paths, self.attributes = sort_paths(*stack_paths(*svgdoc2paths(doc)))
        else:
            self.all_paths, self.attributes = sort_paths(*svgdoc2paths(doc))

        if root_width is not None:
            root_width = root_width.value
            if root_width.find("mm") > 0:
                root_width = float(root_width.replace("mm", ""))
            elif root_width.find("in") > 0:
                root_width = float(root_width.replace("in", "")) * 25.4
            elif root_width.find("px") > 0:
                root_width = float(root_width.replace("px", "")) * 0.264583333
            elif root_width.find("pt") > 0:
                root_width = float(root_width.replace("pt", "")) * 0.264583333
            else:
                root_width = float(root_width)
        size = 4*25.4
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
            bbox = overall_bbox(self.all_paths)
            width = bbox[1] - bbox[0]
            height = bbox[3] - bbox[2]

        if width > height:
            self.scale = size / width
        else:
            self.scale = size / height
        self.generate_pattern()

    def add_block(self, clear=True):
        if len(self.stitches) == 0:
            print("got no stitches in add block!")
        if self.last_color is not None:
            block = Block(stitches=self.stitches, color=self.last_color)
            self.pattern.add_block(block)
        else:
            print("last color was none, not adding the block")
        if clear:
            self.stitches = []

    def generate_pattern(self):
        # cut the paths by the paths above
        if self.fill:
            self.all_paths, self.attributes = stack_paths(self.all_paths, self.attributes)

        for k, v in enumerate(self.attributes):
            paths = self.all_paths[k]

            # first, look for the color from the fill
            # if fill is false, change the attributes so that the fill is none but the
            # stroke is the fill (if not set)
            self.fill_color = get_color(v, "fill")
            self.stroke_color = get_color(v, "stroke")
            stroke_width = get_stroke_width(v, self.scale)
            if not self.fill:
                if not self.stroke_color:
                    self.stroke_color = self.fill_color
                stroke_width = stroke_width if stroke_width != minimum_stitch \
                    else minimum_stitch * 3.0
                self.fill_color = None
            if self.fill_color is None and self.stroke_color is None:
                self.fill_color = [0, 0, 0]
            # if both the fill color and stroke color are none,
            if self.fill_color is not None:
                if len(self.pattern.blocks) == 0 and self.fill_color is not None:
                    self.pattern.add_block(Block([Stitch(["JUMP"], 0, 0)], color=self.fill_color))
                self.switch_color(self.fill_color)
                full_path = Path(*paths)
                if fill_method == "polygon":
                    if not full_path.iscontinuous():
                        self.fill_polygon(make_continuous(full_path))
                    else:
                        self.fill_polygon(paths)
                elif fill_method == "grid":
                    self.fill_grid(paths)
                elif fill_method == "scan":
                    self.fill_scan(paths)
                elif fill_method == "voronoi":
                    self.fill_voronoi(paths)
                self.last_color = self.fill_color
            if self.fill_color is not None:
                self.add_block()
            # then do the stroke
            if self.stroke_color is None:
                continue
            self.switch_color(self.stroke_color)
            paths = self.generate_stroke_width(paths, stroke_width)
            self.generate_straight_stroke(paths)
            if len(self.stitches) > 0:
                self.last_color = self.stroke_color

        if not self.fill:
            self.add_block()

        if len(self.pattern.blocks) > 0 and len(self.pattern.blocks[-1].stitches) > 0:
            last_stitch = self.pattern.blocks[-1].stitches[-1]
            self.pattern.add_block(
                Block(stitches=[Stitch(["END"], last_stitch.x, last_stitch.y)],
                      color=self.pattern.blocks[-1].color))

    def generate_stroke_width(self, paths, stroke_width):
        # paths = remove_close_paths(paths)
        # how many times can the minimum_stitch fit in the stroke width?
        # if it is greater 1, duplicate the stitch offset by the minimum stitch
        if stroke_width / minimum_stitch > 1.:
            new_paths = []
            for i in range(0, int(stroke_width / minimum_stitch)):
                for path in paths:
                    if i == 0:
                        new_paths.append(path)
                        continue
                    # what is the broad angle of the path? (used to determine the
                    # perpendicular angle to translate the path by)
                    num_norm_samples = 10.0
                    diff = average([path.normal(t / num_norm_samples)
                                    for t in range(int(num_norm_samples))])

                    diff *= -1 if i % 2 == 0 else 1
                    diff *= ceil(i / 2.0) * minimum_stitch / 2.0

                    # if i is odd, translate up/left, if even, translate down/right
                    new_paths.append(path.translated(diff))

            return new_paths

    def switch_color(self, new_color):
        if self.last_color is None or self.last_color == new_color or len(
                self.stitches) == 0:
            return
        self.add_block(clear=False)
        to = self.stitches[-1]
        block = Block(stitches=[Stitch(["TRIM"], to.x, to.y)],
                      color=self.last_color)
        self.pattern.add_block(block)
        block = Block(stitches=[Stitch(["COLOR"], to.x, to.y)],
                      color=new_color)
        self.pattern.add_block(block)
        self.stitches = []

    def generate_straight_stroke(self, paths):
        for i, path in enumerate(paths):
            if path.length() == 0:
                continue
            to = Stitch(["STITCH"], path.start.real * self.scale,
                        path.start.imag * self.scale, color=self.stroke_color)
            self.stitches.append(to)
            num_segments = ceil(path.length() / minimum_stitch)
            for seg_i in range(int(num_segments + 1)):
                control_stitch = Stitch(["STITCH"],
                                        path.point(seg_i / num_segments).real * self.scale,
                                        path.point(seg_i / num_segments).imag * self.scale,
                                        color=self.stroke_color)
                self.stitches.append(control_stitch)
            # if the next stitch doesn't start at the end of this stitch, add that one as
            # well
            end_stitch = Stitch(["STITCH"], path.end.real * self.scale,
                                path.end.imag * self.scale, color=self.stroke_color)
            if i != len(paths) - 1:
                if path.end != paths[i + 1].start:
                    self.stitches.append(end_stitch)
            else:
                self.stitches.append(end_stitch)

    def fill_polygon(self, paths):
        rotated = 0
        fudge_factor = 0.03
        while len(paths) > 2:
            if len(paths) < 4:
                self.fill_triangle(paths, color="red")
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
                has_intersection = len(
                    [1 for line in paths if len(line.intersect(test_line1)) > 0]) > 0

            if not path1_is_contained_in_path2(test_line1,
                                               comparison_path) or has_intersection:
                shapes = [[comparison_path, "none", "blue"],
                          [test_line1, "none", "black"]]
                write_debug("anim", shapes)
                # rotate the paths
                paths = paths[1:] + [paths[0]]
                rotated += 1
                if rotated >= len(paths):
                    print("failed to rotate into a concave path -> ",
                          (test_line1.start.real, test_line1.start.imag),
                          (test_line1.end.real, test_line1.end.imag),
                          [(p.start.real, p.start.imag) for p in paths])
                    return
                continue
            side = shorter_side(paths)

            test_line2 = Line(start=paths[1].start, end=paths[2].end)
            test_line2 = Line(start=test_line2.point(fudge_factor),
                              end=test_line2.point(1 - fudge_factor))
            test_line3 = Line(start=paths[-1 + side].end,
                              end=paths[(3 + side) % len(paths)].start)
            test_line3 = Line(start=test_line3.point(fudge_factor),
                              end=test_line3.point(1 - fudge_factor))

            num_intersections = []
            for path in comparison_path:
                if test_line3.length() == 0:
                    print("test line 3 is degenerate!")
                num_intersections += test_line3.intersect(path)
                num_intersections += test_line2.intersect(path)

            rect_not_concave = not path1_is_contained_in_path2(test_line2,
                                                               comparison_path)

            # test for concavity. If concave, fill as triangle
            if is_concave(paths) or len(num_intersections) > 0 or rect_not_concave:
                self.fill_triangle(paths, color="blue")
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
                side, side2 = self.fill_trap(paths)
                if side:
                    paths = paths[1:] + [paths[0]]
                shapes = [[Path(*paths), "none", "black"]]
                to_remove = []
                to_remove.append(paths.pop(0))
                to_remove.append(paths.pop(0))
                to_remove.append(paths.pop(0))
                # if the trap was stitched in the vertical (perpendicular to the
                # stitches), don't remove that segment
                linecolors = ["blue", "purple", "pink"]
                for i, shape in enumerate(to_remove):
                    shapes.append([shape, "none", linecolors[i]])
                closing_line = Line(start=paths[-1].end, end=paths[0].start)
                shapes.append([closing_line, "none", "green"])
                shapes.append([test_line2, "none", "purple"])
                write_debug("rem", shapes)
                delta = closing_line.length() - (
                    test_line3.length() / (1.0 - 2.0 * fudge_factor))
                if abs(delta) > 1e-14:
                    print("closing line different than test!", side, test_line3,
                          closing_line)
            rotated = 0
            if paths[-1].end != paths[0].start:
                # check for intersections
                closing_line = Line(start=paths[-1].end, end=paths[0].start)
                paths.insert(0, closing_line)
            else:
                print("removed paths but they connected anyway")

    def fill_shape(self, side1, side2, paths, shapes):
        if paths[side1].length() == 0:
            return
        increment = 3 * minimum_stitch / paths[side1].length()
        current_t = 0
        # make closed shape

        filled_paths = [paths[side1], paths[side2]]
        if filled_paths[0].end != filled_paths[1].start:
            filled_paths.insert(1, Line(start=filled_paths[0].end,
                                        end=filled_paths[1].start))
        if filled_paths[0].start != filled_paths[-1].end:
            filled_paths.append(Line(start=filled_paths[-1].end,
                                     end=filled_paths[0].start))
        while current_t < 1.0 - increment * 0.5:
            point1 = paths[side1].point(current_t)
            point2 = paths[side2].point(1 - (current_t + 0.5 * increment))
            point3 = paths[side1].point(current_t + increment)
            to = Stitch(["STITCH"], point1.real * self.scale,
                        point1.imag * self.scale,
                        color=self.fill_color)
            self.stitches.append(to)
            to = Stitch(["STITCH"], point2.real * self.scale,
                        point2.imag * self.scale,
                        color=self.fill_color)
            self.stitches.append(to)
            current_t += increment
            to = Stitch(["STITCH"], point3.real * self.scale,
                        point3.imag * self.scale,
                        color=self.fill_color)
            self.stitches.append(to)
        shapes.append([paths[side1], "none", "orange"])
        shapes.append([paths[side2], "none", "red"])
        return shapes

    def fill_grid(self, paths):
        grid = Grid(paths)

        draw_fill(grid, paths)
        # need to find the next location to stitch to. It needs to zig-zag, so we need to
        # keep a record of what direction it was going in
        going_east = True

        rounds = 1
        num_empty = count_empty()
        while num_empty > 0:
            curr_pos = find_upper_corner()
            to = Stitch(["STITCH"], curr_pos.real * self.scale,
                        curr_pos.imag * self.scale,
                        color=self.fill_color)
            self.stitches.append(to)
            blocks_covered = int(maximum_stitch / minimum_stitch)
            while grid.grid_available(curr_pos):
                for i in range(0, blocks_covered):
                    sign = 1.0 if going_east else -1.0
                    test_pos = curr_pos + sign * i * minimum_stitch
                    if not grid.grid_available(test_pos):
                        break
                    else:
                        next_pos = test_pos + 1j * minimum_stitch
                going_east = not going_east
                to = Stitch(["STITCH"], next_pos.real * self.scale,
                            next_pos.imag * self.scale,
                            color=self.fill_color)
                self.stitches.append(to)
                curr_pos = next_pos
            draw_fill(grid, paths)
            new_num_empty = count_empty()
            if new_num_empty == num_empty:
                print("fill was not able to fill any parts of the grid!")
                break
            else:
                num_empty = new_num_empty
            rounds += 1

    def fill_scan(self, paths):
        lines = scan_lines(paths)
        self.attributes = [{"stroke": self.fill_color} for i in range(len(lines))]
        lines, self.attributes = sort_paths(lines, self.attributes)
        if isinstance(lines, list):
            if len(lines) == 0:
                return
            start_point = lines[0].start
        else:
            start_point = lines.start
        to = Stitch(["STITCH"], start_point.real * self.scale,
                    start_point.imag * self.scale, color=self.fill_color)
        self.stitches.append(to)

        for line in lines:
            to = Stitch(["STITCH"], line.start.real * self.scale,
                        line.start.imag * self.scale, color=self.fill_color)
            self.stitches.append(to)
            to = Stitch(["STITCH"], line.end.real * self.scale,
                        line.end.imag * self.scale, color=self.fill_color)
            self.stitches.append(to)

    def cross_stitch_to_pattern(self, _image):
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
        self.all_paths = paths
        self.attributes = attributes
        self.scale = 1.0
        self.generate_pattern()

    def fill_voronoi(self, paths):
        points = []
        for path in paths:
            num_stitches = 100.0 * path.length() / maximum_stitch
            ppoints = [path.point(i / num_stitches) for i in range(int(num_stitches))]
            for ppoint in ppoints:
                points.append([ppoint.real, ppoint.imag])
            points.append([path.end.real, path.end.imag])
        vor = Voronoi(points)
        vertices = vor.vertices

        pxs = [x[0] for x in points]
        pys = [-x[1] for x in points]
        if PLOTTING:
            plt.plot(pxs, pys)
        # restrict the points to ones within the shape
        vertices = [x for i, x in enumerate(vertices)
                    if path1_is_contained_in_path2(Line(end=x[0] + x[1] * 1j,
                                                        start=x[0] + 0.01 + x[
                                                                                1] * 1j),
                                                   Path(*paths))]
        # now sort the vertices. This is close but not quite what is being done in
        # sort_paths
        new_vertices = []
        start_location = points[0]
        while len(vertices) > 0:
            vertices = sorted(vertices,
                              key=lambda x: (start_location[0] - x[0]) ** 2
                                            + (start_location[1] - x[1]) ** 2)
            new_vertices.append(vertices.pop(0))
            start_location = new_vertices[-1]
        vertices = new_vertices
        # now smooth out the vertices
        vertices = [[[x[0] for x in vertices[i:i + 3]],
                     [x[1] for x in vertices[i:i + 3]]]
                    for i in range(0, len(vertices) - 3)]
        vertices = [[average(x[0]), average(x[1])] for x in vertices]
        # we want each vertice to be about equidistant
        vertices = make_equidistant(vertices, minimum_stitch / 2.0)
        xs = [x[0] for x in vertices]
        ys = [-x[1] for x in vertices]
        if PLOTTING:
            plt.plot(xs, ys, 'r-')
        stitchx = [vertices[0][0]]
        stitchy = [vertices[0][1]]

        # make spines
        for i in range(len(vertices) - 1):
            intersections = perpendicular(vertices[i][0] + vertices[i][1] * 1j,
                                          vertices[i + 1][0] + vertices[i + 1][
                                              1] * 1j,
                                          Path(*paths))
            diff = abs(intersections[0] - intersections[1])
            if diff > 9:
                continue
            stitchx.append(intersections[0].real)
            stitchy.append(-intersections[0].imag)
            stitchx.append(intersections[1].real)
            stitchy.append(-intersections[1].imag)
        for i in range(len(stitchx)):
            to = Stitch(["STITCH"], stitchx[i] * self.scale,
                        -stitchy[i] * self.scale, color=self.fill_color)
            self.stitches.append(to)
        if PLOTTING:
            plt.plot(stitchx, stitchy, 'g-')
            plt.xlim(min(pxs), max(pxs))
            plt.ylim(min(pys), max(pys))
            # plt.show()

    def fill_trap(self, paths, color="gray"):
        side = shorter_side(paths)
        shapes = [[Path(*paths), "none", "black"],
                  [Path(*paths[side:side + 3]), color, "none"]]
        side2 = side + 2
        shapes = self.fill_shape(side, side2, paths, shapes)
        write_debug("fill", shapes)
        return side, side2

    def fill_triangle(self, paths, color="green"):
        triangle_sides = [paths[0], paths[1],
                          Line(start=paths[2].start, end=paths[0].start)]
        shapes = [[Path(*paths), "none", "black"],
                  [Path(*triangle_sides), color, "none"]]
        lengths = [p.length() for p in triangle_sides]
        side1 = argmax(lengths)
        lengths[side1] = 0
        side2 = argmax(lengths)
        shapes = self.fill_shape(side1, side2, triangle_sides, shapes)
        write_debug("fill", shapes)


if __name__ == "__main__":
    start = time()
    args = parser.parse_args()
    filename = args.filename
    dig = Digitizer(filename=filename, fill=args.fill)
    end = time()
    print("digitizer time: %s" % (end - start))
    try:
        measure_density(dig.pattern)
    except ValueError as e:
        pass
    pattern = de_densify(dig.pattern)
    measure_density(pattern)
    pattern_to_csv(pattern, join(OUTPUT_DIRECTORY, filename + ".csv"))
    pattern_to_svg(pattern, join(OUTPUT_DIRECTORY, filename + ".svg"))
    pes_filename = join(OUTPUT_DIRECTORY, filename + ".pes")
    bef = BrotherEmbroideryFile(pes_filename)
    bef.write_pattern(pattern)
    upload(pes_filename)