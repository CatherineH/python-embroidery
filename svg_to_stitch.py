from os.path import join, dirname, abspath

from svgpathtools import svg2paths
from sys import argv
from webcolors import hex_to_rgb
import csv
from subprocess import call


paths, attributes = svg2paths(argv[1])

xlims = [None, None]
ylims = [None, None]

for path in paths:
    for segment in path:
        # xlimits
        if segment.start.real >= xlims[1]:
            xlims[1] = segment.start.real
        if segment.end.real >= xlims[1]:
            xlims[1] = segment.end.real
        if xlims[0] is None:
            xlims[0] = segment.start.real
        if segment.start.real < xlims[0]:
            xlims[0] = segment.start.real
        if segment.end.real < xlims[0]:
            xlims[0] = segment.end.real
        # y limits
        if segment.start.imag >= ylims[1]:
            ylims[1] = segment.start.imag
        if segment.end.imag >= ylims[1]:
            ylims[1] = segment.end.imag
        if ylims[0] is None:
            ylims[0] = segment.start.imag
        if segment.start.imag < ylims[0]:
            ylims[0] = segment.start.imag
        if segment.end.imag < ylims[0]:
            ylims[0] = segment.end.imag

print(ylims, xlims)

width = abs(xlims[0] - xlims[1])
height = abs(ylims[0] - ylims[1])

# this script is in mms. The assumed minimum stitch width is 1 mm, the maximum width is 5 mm
minimum_stitch = 5.0
maximum_stitch = 5.0

# The maximum size is 4 inches
size = 4.0 * 25.4
if width > height:
    scale = size / width
else:
    scale = size / height


class Stitch(object):
    color = None
    x = []
    y = []


def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


stitches = []

for k, v in enumerate(attributes):
    to = Stitch()
    # first, look for the color from the fill
    if v['style'].find('fill:') >= 0:
        to.color = v['style'].split('fill:')[1].split(";")[0]
    else:
        # the color is black
        to.color = '#000000'
    for i, path in enumerate(paths[k]):
        t = 0.0
        if path.length() == 0:
            continue
        while t <= 1.0:
            to.x.append(path.point(t).real * scale)
            to.y.append(height*scale-path.point(t).imag * scale)
            t += minimum_stitch / path.length()
    stitches.append(to)

# get a unique list of thread colors
colors = set([to.color for to in stitches])

# start writing the output file
of = open(argv[1].replace('.svg', '.csv'), 'w')
csvwriter = csv.writer(of, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
intro_lines = ["Embroidermodder 2 CSV Embroidery File",
               "http://embroidermodder.github.io",
               "Generated from SVG using python-embroidery",
               "https://github.com/CatherineH/python-embroidery"]
for line in intro_lines:
    csvwriter.writerow(["#", line])
csvwriter.writerow(["#", "[VAR_NAME]", "[VAR_VALUE]"])

vars = {"STITCH_COUNT": sum([len(st.x) - 1 for st in stitches]),
        "THREAD_COUNT": len(colors), "EXTENTS_LEFT": xlims[0] * scale,
        "EXTENTS_TOP": ylims[0] * scale, "EXTENTS_RIGHT": xlims[1] * scale,
        "EXTENTS_BOTTOM": ylims[1] * scale, "EXTENTS_WIDTH": width * scale,
        "EXTENTS_HEIGHT": height * scale}

for varkey in vars:
    csvwriter.writerow([">", varkey + ":", vars[varkey]])

csvwriter.writerow(["#", "[THREAD_NUMBER]", "[RED]", "[GREEN]", "[BLUE]", "[DESCRIPTION]",
                    "[CATALOG_NUMBER]"])
for i, color in enumerate(colors):
    csvwriter.writerow(["$", i] + list(hex_to_rgb(color)) + ["(null)", "(null)"])

csvwriter.writerow(["#", "[STITCH_TYPE]", "[X]", "[Y]"])
# stitches need to be sorted by color in order to minimize thread changes
for i, color in enumerate(colors):
    stitches_by_col = [st for st in stitches if st.color == color]
    if i != 0:
        # add a thread change
        csvwriter.writerow(["*", "TRIM", stitches_by_col[0].x[0],stitches_by_col[0].y[0]])
        csvwriter.writerow(["*", "COLOR", stitches_by_col[0].x[0], stitches_by_col[0].y[0]])
    for stitch in stitches_by_col:
        csvwriter.writerow(["*", "JUMP", stitch.x[0], stitch.y[0]])
        for j in range(len(stitch.x)):
            csvwriter.writerow(["*", "STITCH", stitch.x[j], stitch.y[j]])
    # if on last color, add end statement
    if i == len(colors)-1:
        csvwriter.writerow(["*", "END", stitches_by_col[-1].x[-1], stitches_by_col[-1].x[-1]])

# convert the file to PES format
full_filename = abspath(argv[1]).replace(".svg", ".csv")
print("CSV has been generated: ", full_filename)
try:
    call(["libembroidery-convert", full_filename, full_filename.replace(".csv", ".pes")])
except OSError as e:
    print("Could not find libembroidery-convert. Did you install embroiderymodder and add libembroidery-convert to your path?")