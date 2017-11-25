import re
from svgpathtools import svg2paths
from bezier import Curve
from numpy import asfortranarray, interp
from sys import argv

paths, attributes = svg2paths(argv[1])

xlims = [None, None]
ylims = [None, None]

for path in paths:
    # xlimits
    if path.start.real >= xlims[1]:
        xlims[1] = path.start.real
    if path.end.real >= xlims[1]:
        xlims[1] = path.end.real
    if xlims[0] is None:
        xlims[0] = path.start.real
    if path.start.real < xlims[0]:
        xlims[0] = path.start.real
    if path.end.real < xlims[0]:
        xlims[0] = path.end.real
    # y limits
    if path.start.imag >= ylims[1]:
        ylims[1] = path.start.imag
    if path.end.imag >= ylims[1]:
        ylims[1] = path.end.imag
    if ylims[0] is None:
        ylims[0] = path.start.imag
    if path.start.imag < ylims[0]:
        ylims[0] = path.start.imag
    if path.end.imag < ylims[0]:
        ylims[0] = path.end.imag

width = abs(xlims[0]-xlims[1])
height = abs(ylims[0]-ylims[1])


# this script is in mms. The assumed minimum stitch width is 1 mm, the maximum width is 5 mm

# The maximum size is 4 inches
size = 4.0*25.4
if width > height:
    scale = size/width
else:
    scale = size/height


"#","[VAR_NAME]","[VAR_VALUE]"
">","STITCH_COUNT:","137"
">","THREAD_COUNT:","5"
">","EXTENTS_LEFT:","-11.917100"
">","EXTENTS_TOP:","-37.133000"
">","EXTENTS_RIGHT:","13.644200"
">","EXTENTS_BOTTOM:","4.145080"
">","EXTENTS_WIDTH:","25.561300"
">","EXTENTS_HEIGHT:","41.278080"


'''

nodes1 = asfortranarray([[0.0, 0.0],[0.5, 1.0],[1.0, 0.0],])

curve1 = Curve(nodes1, degree=2)
for i in range(10):
    print(i/10.0, curve1.evaluate(i/10.0))
'''


class Thread(object):
    color = None
    x = None
    y = None


def distance(point1, point2):
    return ((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5


for k, v in enumerate(attributes):
    to = Thread()
    # first, look for the color from the fill
    if v['style'].find('fill:') >= 0:
        to.color = v['style'].split('fill:')[1].split(";")[0]
    else:
        # the color is black
        to.color = '#000000'
    # scan the path left to right
    path_parts = []
    path_type = ""
    path = v['d'].lower()
    while len(path) > 1:
        parts = re.match('([mcl])([\d\.\ \,\-]+)([\S\s]+)', path)
        if parts is None:
            break
        newpath = parts.group(3)
        if path == newpath:
            print("failure! ", parts.groups())
            break
        path_parts.append([parts.group(1), [[float(p) for p in c.split(',')] for c in parts.group(2).strip().split(' ')]])
        path = newpath
    # add the first move to point to the subsequent points
    point = [None, None]
    for path_part in path_parts:
        if path_part[0] == 'm':
            point = path_part[1]
            print(path_part[1])
        else:
            
'''

    for _char in v['d']:
        if _char.lower() in ['m', 'c', 'l', 'z']:
            if path_type != "" and len(path.strip()) != 0:
                path_parts.append((path_type, [[scale*float(x) for x in p.split(',')] for p in path.strip().split(' ')]))
            path_type = _char.lower()
            path = ""
        else:
            path += _char
    to.x = [path_parts[0][1][0][0]]
    to.y = [path_parts[0][1][0][1]]
    for i in range(1, len(path_parts)):
        _distance = distance(path_parts[i-1][1][-1], path_parts[i][1][-1])
        if _distance < 1.0:
            to.x.append([path_parts[i][1][-1][0]])
            to.y.append([path_parts[i][1][-1][0]])
        elif path_parts[i][0].lower() == 'c':
            nodes = asfortranarray([path_parts[i-1][1][-1]]+path_parts[i][1])
            curve = Curve(nodes, degree=2)
            currx = path_parts[i-1][1][-1][0]+0.7
            while currx <= path_parts[i][1][-1][0]:
                to.x.append(currx)
                to.y.append(curve.evaluate(currx))
                currx += 0.7
        elif path_parts[i][0].lower() == 'l':
            currx = path_parts[i-1][1][-1][0]+0.7
            while currx <= path_parts[i][1][-1][0]:
                y = interp(currx, [path_parts[i-1][1][-1][0], path_parts[i][1][-1][0]],
                       [path_parts[i-1][1][-1][1], path_parts[i][1][-1][1]])
                to.x.append(currx)
                to.y.append(y)
                currx += 0.7
        else:
            print(path_parts[i][0], v['d'])
    #print(k, v['d'])  # print d-string of k-th path in SVG
'''