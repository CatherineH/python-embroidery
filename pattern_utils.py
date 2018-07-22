"""A set of functionality to apply to completed patterns"""
import csv
from math import ceil
from time import time

from block import Block
from configure import minimum_stitch, PLOTTING, OUTPUT_DIRECTORY
from os.path import join
from pattern import Pattern
from stitch import Stitch

from svgpathtools import Line, Path, wsvg
from svgutils import overall_bbox
from svgwrite import Drawing, rgb

if PLOTTING:
    import matplotlib.pyplot as plt
else:
    plt = None


# spiral around a point until you find the next available location
class NextAvailableGrid(object):
    def __init__(self, i, j):
        self.i = i # i is the x coordinate
        self.j = j # j is the y coordinate
        self.direction = "left"
        self.stepsize = 1
        self.current_step = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        directions = ["left", "down", "right", "up"] # go counter-clockwise
        # do the transforms
        if self.direction == "left":
            self.i += 1
        if self.direction == "down":
            self.j += 1
        if self.direction == "right":
            self.i -= 1
        if self.direction == "up":
            self.j -= 1
        self.current_step += 1
        if self.current_step == self.stepsize:
            self.direction = directions[(directions.index(self.direction) + 1)
                                        % len(directions)]
            self.current_step = 0
            if self.direction in ["right", "left"]:
                self.stepsize += 1

        return self.i, self.j


def pattern_to_svg(pattern, filename):
    if isinstance(filename, str) or isinstance(filename, unicode):
        output_file = open(filename, "wb")
    else:
        output_file = filename
    paths = []
    colors = []
    scale_factor = 0.1 # scale from cm to mm from pes
    for block in pattern.blocks:
        block_paths = []
        last_stitch = None
        for stitch in block.stitches:
            if "JUMP" in stitch.tags:
                last_stitch = stitch
                continue
            if last_stitch is None:
                last_stitch = stitch
                continue
            block_paths.append(Line(start=last_stitch.complex*scale_factor,
                                    end=stitch.complex*scale_factor))
            last_stitch = stitch
        if len(block_paths) > 0:
            colors.append(block.tuple_color)
            paths.append(Path(*block_paths))
    dims = overall_bbox(paths)
    mindim = max(dims[1]-dims[0], dims[3]-dims[2])
    print("in pattern to svg, overallbbox", overall_bbox(paths))
    if len(paths) == 0:
        print("warning: pattern did not generate stitches")
        return
    wsvg(paths, colors, filename = output_file, mindim=mindim)


def csv_to_pattern(filename):
    pattern = Pattern()
    stitches = []
    # read in the stitch file
    last_flag = None
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if len(row) < 4:
                continue
            if row[0] == '*':
                if row[1] != last_flag and last_flag is not None:
                    if len(pattern.blocks) == 0:
                        # prepend the first block with a 0,0 JUMP Stitch
                        stitches.insert(0, Stitch(["JUMP"], 0, 0))
                    pattern.blocks.append(Block(stitches=stitches,
                                                color=pattern.colors[
                                                    pattern.current_color]))
                    stitches = []
                stitches.append(
                    Stitch([row[1]], float(row[2]) * 10.0, -float(row[3]) * 10.0))
                if row[1] == "COLOR":
                    pattern.current_color += 1
                last_flag = row[1]
            elif row[0] == "$":
                pattern.colors.append(row[2:5])

    # add the ending block
    end_stitch = Stitch(["END"], pattern.blocks[-1].stitches[-1].x,
                        pattern.blocks[-1].stitches[-1].y)
    pattern.blocks.append(
        Block(stitches=[end_stitch], color=pattern.colors[pattern.current_color]))
    return pattern


def initialize_grid(pattern):
    if len(pattern.all_stitches) == 0:
        return [[]], 0, 0, 0, 0
    boundx = max([s[0] for s in pattern.all_stitches]), min([s[0] for s in pattern.all_stitches])
    boundy = max([s[1] for s in pattern.all_stitches]), min([s[1] for s in pattern.all_stitches])
    x_bins = int(ceil((boundx[0]-boundx[1])/minimum_stitch))
    y_bins = int(ceil((boundy[0]-boundy[1])/minimum_stitch))
    print(x_bins, y_bins)
    density = [[0 for _ in range(x_bins)] for _ in range(y_bins)]
    return density, boundx, boundy, x_bins, y_bins


def measure_density(pattern):
    density, boundx, boundy, x_bins, y_bins = initialize_grid(pattern)
    for stitch in pattern.all_stitches:
        i = int((stitch[0] - boundx[1]) / minimum_stitch)
        j = int((stitch[1] - boundy[1]) / minimum_stitch)
        density[j][i] += 1
    for i in range(x_bins):
        for j in range(y_bins):
            density[j][i] = density[j][i] # if density[j][i] > 3 else 0
    plot_density(density, pattern)
    for values in density:
        for value in values:
            if value > MAX_STITCHES:
                raise ValueError("stitch density is greater than 3. Check density.png.")


def density_dict_to_list(density):
    max_x = max(density.keys())
    min_x = min(density.keys())
    all_y_keys = []
    for y_slice in density.values():
        all_y_keys += y_slice.keys()
    if len(all_y_keys) == 0:
        return [[]]
    max_y = max(all_y_keys)
    min_y = min(all_y_keys)
    new_density = [[0 for _ in range(min_y, max_y + 1)] for _ in range(min_x, max_x + 1)]
    for x, y_slice in density.items():
        for y, val in density[x].items():
            new_density[x - min_x][y - min_y] = density[x][y]
    return new_density


def plot_density(density, pattern):
    if isinstance(density, dict):
        density = density_dict_to_list(density)

    fig, ax = plt.subplots()
    ax.imshow(density)
    heatmap = ax.pcolor(density, cmap=plt.cm.Blues, vmin=0, vmax=8)
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Stitch density per cell')
    x_stitches = [s[0]/minimum_stitch+1 for s in pattern.all_stitches]
    y_stitches = [s[1]/minimum_stitch+1 for s in pattern.all_stitches]
    plt.plot(x_stitches, y_stitches, 'r--', alpha=0.5)
    fig.savefig(join(OUTPUT_DIRECTORY, 'density_{}.png'.format(time())))  # save the figure to file
    plt.close(fig)


MAX_STITCHES = 3


def de_densify(pattern):
    density, boundx, boundy, x_bins, y_bins = initialize_grid(pattern)
    # convert the density list of lists to a dict
    density = {i: {j: block for j, block in enumerate(density[i])}
               for i in range(len(density))}
    for block_i, block in enumerate(pattern.blocks):
        for stitch_i, stitch in enumerate(block.stitches):
            i = int((stitch.x - boundx[1]) / minimum_stitch)
            j = int((stitch.y - boundy[1]) / minimum_stitch)
            # if there is room for that stitch, continue
            if density[j][i] <= MAX_STITCHES:
                density[j][i] += 1
                continue
            for next_i, next_j in NextAvailableGrid(i, j):
                if density[next_j][next_i] >= MAX_STITCHES:
                    continue
                print("moving stitch from {} {}".format(stitch.x, stitch.y))
                pattern.blocks[block_i].stitches[stitch_i].x = next_i * minimum_stitch + boundx[1]
                pattern.blocks[block_i].stitches[stitch_i].y = next_j * minimum_stitch + \
                                                               boundy[1]
                print("to {} {}".format(pattern.blocks[block_i].stitches[stitch_i].x, pattern.blocks[block_i].stitches[stitch_i].y))
                density[next_j][next_i] += 1
                break
    return pattern


if __name__ == "__main__":
    # make a graph of the next available grid
    num_grid = 8
    spacing = 30

    dwg = Drawing(join(OUTPUT_DIRECTORY, "next_available_grid.svg"),
                  (num_grid*spacing, num_grid*spacing))
    for x in range(num_grid):
        dwg.add(dwg.line(start=(0, x*spacing), end=(num_grid*spacing, x*spacing),
                         stroke=rgb(10, 10, 16, '%')))
        dwg.add(dwg.line(start=(x * spacing, 0), end=(x * spacing, num_grid * spacing),
                         stroke=rgb(10, 10, 16, '%')))
    start_point = [4, 4]

    dwg.add(dwg.rect(insert=(start_point[0]*spacing, start_point[1]*spacing), size=(spacing, spacing),
                     fill=rgb(100, 100, 16, '%')))
    count = 0
    last_point = start_point
    for next_available in NextAvailableGrid(*start_point):
        dwg.add(dwg.line(start=((last_point[0]+0.5)*spacing, (last_point[1]+0.5)*spacing),
                         end=((next_available[0]+0.5)*spacing, (next_available[1]+0.5)*spacing),
                         stroke=rgb(100, 0, 0, '%')))
        last_point = next_available
        count += 1
        if count > 25:
            break
    dwg.save()