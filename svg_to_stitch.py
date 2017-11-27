from os import statvfs, remove
from os.path import join, abspath, basename, getsize, exists
from shutil import copyfile
from xml.etree import ElementTree

import re
from svgpathtools import svg2paths
from sys import argv
import csv
from subprocess import call, check_output

tree = ElementTree.parse(argv[1])
root = tree.getroot()
# make sure the document size is appropriate
assert root.attrib['width'] == '4in'
assert root.attrib['height'] == '4in'

paths, attributes = svg2paths(argv[1])
lims = [float(i) for i in root.attrib['viewBox'].split(" ")]

width = abs(lims[0] - lims[2])
height = abs(lims[1] - lims[3])

# this script is in mms. The assumed minimum stitch width is 1 mm, the maximum width is 5 mm
minimum_stitch = 15.0
maximum_stitch = 15.0

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
    #if k > 0:
    #    continue
    to = Stitch()
    # first, look for the color from the fill
    if v['style'].find('fill:') >= 0:
        to.color = v['style'].split('fill:')[1].split(";")[0]
    else:
        # the color is black
        to.color = '#000000'
    for i, path in enumerate(paths[k]):
        if path.length() == 0:
            continue
        if i > 0:
            t = minimum_stitch / path.length()
        else:
            t = 0.0
        if path.length() == 0:
            continue
        print(type(path))
        while t <= 1.0:
            print(path.point(t).real * scale, height*scale-path.point(t).imag * scale)
            to.x.append(path.point(t).real * scale)
            to.y.append(height*scale-path.point(t).imag * scale)
            if str(type(path)).find("Line") >= 0:
                t += 1.0
                print(t)
            else:
                t += minimum_stitch / path.length()

    stitches.append(to)

# get a unique list of thread colors
colors = list(set([to.color for to in stitches]))
#colors = [colors[0]]
# start writing the output file
csv_filename = argv[1].replace('.svg', '.csv')
if exists(csv_filename):
    # this step is somewhat unnecessary, but I want to be absolutely sure that the new
    # design is being uploaded
    remove(csv_filename)
of = open(csv_filename, 'w')
csvwriter = csv.writer(of, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
intro_lines = ["Embroidermodder 2 CSV Embroidery File",
               "http://embroidermodder.github.io",
               "Generated from SVG using python-embroidery",
               "https://github.com/CatherineH/python-embroidery"]
for line in intro_lines:
    csvwriter.writerow(["#", line])
csvwriter.writerow(["#", "[VAR_NAME]", "[VAR_VALUE]"])

vars = {"STITCH_COUNT": sum([len(st.x) - 1 for st in stitches]),
        "THREAD_COUNT": len(colors), "EXTENTS_LEFT": lims[0] * scale,
        "EXTENTS_TOP": lims[1] * scale, "EXTENTS_RIGHT": lims[2] * scale,
        "EXTENTS_BOTTOM": lims[3] * scale, "EXTENTS_WIDTH": width * scale,
        "EXTENTS_HEIGHT": height * scale}

for varkey in vars:
    csvwriter.writerow([">", varkey + ":", vars[varkey]])

csvwriter.writerow(["#", "[THREAD_NUMBER]", "[RED]", "[GREEN]", "[BLUE]", "[DESCRIPTION]",
                    "[CATALOG_NUMBER]"])
for i, color in enumerate(colors):
    csvwriter.writerow(["$", i+1] + [0, 0, 0] + ["(null)", "(null)"])

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
of.close()

# convert the file to PES format
full_filename = abspath(argv[1]).replace(".svg", ".csv")
pes_filename = full_filename.replace(".csv", ".pes")
if exists(pes_filename):
    remove(pes_filename)
print("CSV has been generated: ", full_filename)
try:
    call(["libembroidery-convert", full_filename, pes_filename])
except OSError as e:
    print("Could not find libembroidery-convert. Did you install embroiderymodder and "
          "add libembroidery-convert to your path?")


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