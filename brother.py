import struct
import csv
from os.path import basename, splitext

from numpy import argmin

pecThreads = [[[0, 0, 0], "Unknown", ""],
              [[14, 31, 124], "Prussian Blue", ""],
              [[10, 85, 163], "Blue", ""],
              [[0, 135, 119], "Teal Green", ""],
              [[75, 107, 175], "Cornflower Blue", ""],
              [[237, 23, 31], "Red", ""],
              [[209, 92, 0], "Reddish Brown", ""],
              [[145, 54, 151], "Magenta", ""],
              [[228, 154, 203], "Light Lilac", ""],
              [[145, 95, 172], "Lilac", ""],
              [[158, 214, 125], "Mint Green", ""],
              [[232, 169, 0], "Deep Gold", ""],
              [[254, 186, 53], "Orange", ""],
              [[255, 255, 0], "Yellow", ""],
              [[112, 188, 31], "Lime Green", ""],
              [[186, 152, 0], "Brass", ""],
              [[168, 168, 168], "Silver", ""],
              [[125, 111, 0], "Russet Brown", ""],
              [[255, 255, 179], "Cream Brown", ""],
              [[79, 85, 86], "Pewter", ""],
              [[0, 0, 0], "Black", ""],
              [[11, 61, 145], "Ultramarine", ""],
              [[119, 1, 118], "Royal Purple", ""],
              [[41, 49, 51], "Dark Gray", ""],
              [[42, 19, 1], "Dark Brown", ""],
              [[246, 74, 138], "Deep Rose", ""],
              [[178, 118, 36], "Light Brown", ""],
              [[252, 187, 197], "Salmon Pink", ""],
              [[254, 55, 15], "Vermillion", ""],
              [[240, 240, 240], "White", ""],
              [[106, 28, 138], "Violet", ""],
              [[168, 221, 196], "Seacrest", ""],
              [[37, 132, 187], "Sky Blue", ""],
              [[254, 179, 67], "Pumpkin", ""],
              [[255, 243, 107], "Cream Yellow", ""],
              [[208, 166, 96], "Khaki", ""],
              [[209, 84, 0], "Clay Brown", ""],
              [[102, 186, 73], "Leaf Green", ""],
              [[19, 74, 70], "Peacock Blue", ""],
              [[135, 135, 135], "Gray", ""],
              [[216, 204, 198], "Warm Gray", ""],
              [[67, 86, 7], "Dark Olive", ""],
              [[253, 217, 222], "Flesh Pink", ""],
              [[249, 147, 188], "Pink", ""],
              [[0, 56, 34], "Deep Green", ""],
              [[178, 175, 212], "Lavender", ""],
              [[104, 106, 176], "Wisteria Violet", ""],
              [[239, 227, 185], "Beige", ""],
              [[247, 56, 102], "Carmine", ""],
              [[181, 75, 100], "Amber Red", ""],
              [[19, 43, 26], "Olive Green", ""],
              [[199, 1, 86], "Dark Fuschia", ""],
              [[254, 158, 50], "Tangerine", ""],
              [[168, 222, 235], "Light Blue", ""],
              [[0, 103, 62], "Emerald Green", ""],
              [[78, 41, 144], "Purple", ""],
              [[47, 126, 32], "Moss Green", ""],
              [[255, 204, 204], "Flesh Pink", ""],
              [[255, 217, 17], "Harvest Gold", ""],
              [[9, 91, 166], "Electric Blue", ""],
              [[240, 249, 112], "Lemon Yellow", ""],
              [[227, 243, 91], "Fresh Green", ""],
              [[255, 153, 0], "Orange", ""],
              [[255, 240, 141], "Cream Yellow", ""],
              [[255, 200, 200], "Applique", ""]]

imageWithFrame = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]


class Pattern():
    def __init__(self):
        self.blocks = []
        self.stitches = []
        self.colors = []
        self.current_color = 0
        self.image = imageWithFrame

    def add_block(self, block):
        # does some validation on adding a block
        if len(block.stitches) == 0:
            raise ValueError("adding block with no stitches! %s" % block)
        self.blocks.append(block)

    @property
    def bounds(self):
        return calc_bounding_box(self.blocks)

    @property
    def thread_count(self):
        # go through the list of blocks, tally up the number of times the thread color
        # changes
        return len([i for i in range(1, len(self.blocks))
                    if self.blocks[i - 1].color != self.blocks[i].color]) + 1

    @property
    def thread_colors(self):
        blocks_with_stitches = [block for block in self.blocks if len(block.stitches) > 0]
        if len(blocks_with_stitches) != len(self.blocks):
            raise ValueError("This pattern has a block with no stitches!")
        # return set([block.color for block in self.blocks])
        # look for trim, or end not color, to signify the end of the pattern
        return [block.color for block in blocks_with_stitches
                if "TRIM" in block.stitches[0].tags or "END" in block.stitches[0].tags]

    def __str__(self):
        output = ""
        for i, block in enumerate(self.blocks):
            output += "Block Number: %s, color %s, number of stitches: %s\n" % \
                      (i, block.color, block.num_stitches)
        return output

    def __repr__(self):
        return self.__str__()


def nearest_color(in_color):
    # if in_color is an int, assume that the nearest color has already been find
    if isinstance(in_color, int):
        return in_color
    if "red" in dir(in_color):
        in_color = [in_color.red, in_color.green, in_color.blue]
    in_color = [float(p) for p in in_color]
    return argmin(
        [sum([(in_color[i] - color[0][i]) ** 2 for i in range(3)]) for color in pecThreads
         if color[1] != 'Unknown']) + 1


class Stitch:
    def __init__(self, tags=[], xx=0.0, yy=0.0, color=0):
        self.tags = tags
        self.xx = xx
        self.yy = yy
        self.color = color

    def __str__(self):
        return "{} {} {}\n".format(self.xx, self.yy, self.tags[0])


class Block:
    # a block is a contiguous set of stitches
    def __init__(self, stitches=[], color=[0, 0, 0]):
        self.stitches = stitches
        self.color = nearest_color(color)
        if not isinstance(self.color, int):
            raise ValueError(
                "block was instantiated with something that was not a color: %s %s" % (
                    self.color, color))

    @property
    def stitch_type(self):
        return 1 if "JUMP" in self.stitches[0].tags else 0

    @property
    def num_stitches(self):
        return len(self.stitches)  # - self.stitch_type

    def __str__(self):
        if self.num_stitches == 0:
            return "Block has no stitches"
        output = "stitch_type: {}, color: {}, num_stitches: {}\n".format(self.stitch_type,
                                                                         self.color,
                                                                         self.num_stitches)
        for stitch in self.stitches:
            output += str(stitch)
        return output

    def __repr__(self):
        return self.__str__()


class BoundingBox:
    def __init__(self, left=0, top=0, right=0, bottom=0):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    def __str__(self):
        return "width: %s, height: %s" % (self.width, self.height)

    def __repr__(self):
        return self.__str__()
    

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
    end_stitch = Stitch(["END"], pattern.blocks[-1].stitches[-1].xx,
                        pattern.blocks[-1].stitches[-1].yy)
    pattern.blocks.append(
        Block(stitches=[end_stitch], color=pattern.colors[pattern.current_color]))
    return pattern


def calc_bounding_box(blocks):
    bounding_rect = BoundingBox()
    for block in blocks:
        stitches = block.stitches
        for stitch in stitches:
            if "TRIM" in stitch.tags:
                continue
            bounding_rect.left = min(bounding_rect.left, stitch.xx)
            bounding_rect.top = min(bounding_rect.top, stitch.yy)
            bounding_rect.right = max(bounding_rect.right, stitch.xx)
            bounding_rect.bottom = max(bounding_rect.bottom, stitch.yy)
    return bounding_rect


class BrotherEmbroideryFile():
    def __init__(self, filename="output.pes", filehandle=None):
        if filehandle is None:
            self.fh = open(filename, "wb")
        else:
            self.fh = filehandle
        self.filename = filename

        self.verbose = False
        self.output = ''

    def write_pattern(self, pattern):
        self.output += self.write_int(0x00, width=4)

        for _ in range(3):
            self.output += self.write_int(0x01)
        self.output += self.write_int(0x00)
        self.output += self.write_int(0x00)
        self.output += self.write_int(0x07)
        self.c_emb_one(pattern)
        if self.verbose:
            for b in pattern.blocks:
                print(b)

        self.c_sew_seg(pattern)

        self.output += "#PES0001"
        pec_location = len(self.output)

        self.output = self.output[:0x08] + self.write_int(pec_location,
                                                          width=4) + self.output[
                                                                     0x08 + 4:]

        self.write_pec_stitches(pattern)
        self.fh.write(self.output)
        self.fh.close()

    def write_int(self, val, width=2, LE=True):
        output = ''
        val = int(round(val))
        for i in range(width):
            offset = i * 8 if LE else (width - i - 1) * 8
            output += chr((val >> offset) & 0xFF)
        return output

    def write_float(self, val):
        for b in bytearray(struct.pack("f", val)):
            self.output += chr(b)

    def write_stitch(self, stitch, bounds):
        self.output += self.write_int(stitch.xx - bounds.left)
        self.output += self.write_int(stitch.yy + bounds.top)

    def write_image(self, image):
        for i in range(38):
            for j in range(6):
                offset = j * 8
                output = 0
                for k in range(8):
                    output += (image[i][offset + k] != 0) << k
                self.output += chr(output)

    def encode_jump(self, x, tags):
        outputVal = abs(x) & 0x7FF
        orPart = 0x80

        if "TRIM" in tags:
            orPart |= 0x20
        if "JUMP" in tags:
            orPart |= 0x10

        if x < 0:
            outputVal = x + 0x1000 & 0x7FF
            outputVal |= 0x800
        self.output += chr(((outputVal >> 8) & 0x0F) | orPart)
        self.output += chr(outputVal & 0xFF)

    def c_emb_one(self, pattern):
        self.output += "CEmbOne"
        for _ in range(8):
            self.output += self.write_int(0x00)
        self.write_float(1.0)
        self.write_float(0.0)
        self.write_float(0.0)
        self.write_float(1.0)
        hoopHeight = 1800
        hoopWidth = 1300
        self.write_float((pattern.bounds.width - hoopWidth) / 2.0)
        self.write_float((pattern.bounds.height + hoopHeight) / 2.0)
        self.output += self.write_int(0x01)
        self.output += self.write_int(0x00)
        self.output += self.write_int(0x00)
        self.output += self.write_int(pattern.bounds.width)
        self.output += self.write_int(pattern.bounds.height)

        for _ in range(8):
            self.output += chr(0)

    def c_sew_seg(self, pattern):
        self.output += self.write_int(len(pattern.blocks))
        self.output += self.write_int(0xFFFF)
        self.output += self.write_int(0x00)
        self.output += self.write_int(0x07)
        self.output += "CSewSeg"
        colorInfo = []

        for i, block in enumerate(pattern.blocks):
            self.output += self.write_int(block.stitch_type)
            self.output += self.write_int(block.color)
            if block.color not in [c[1] for c in colorInfo]:
                colorInfo.append([i, block.color])
                self.output += self.write_int(block.num_stitches)
            for stitch_i in range(0, len(block.stitches)):
                self.write_stitch(block.stitches[stitch_i], pattern.bounds)
            if i < len(pattern.blocks) - 1:
                self.output += self.write_int(0x8003)
            self.output += self.write_int(len(colorInfo))
        for i in range(len(colorInfo)):
            self.output += self.write_int(colorInfo[i][0])
            self.output += self.write_int(colorInfo[i][1])

        self.output += self.write_int(0, width=4)

    def write_pec_stitches(self, pattern):
        self.output += "LA:"
        pes_name = splitext(basename(self.filename))[0]
        while len(pes_name) < 16:
            pes_name = pes_name + chr(0x20)
        pes_name = pes_name[0:16]

        print(self.filename, pes_name)
        self.output += str(pes_name)
        self.output += chr(0x0D)
        for _ in range(12):
            self.output += chr(0x20)
        self.output += chr(0xFF)
        self.output += chr(0x00)
        self.output += chr(0x06)
        self.output += chr(0x26)
        for _ in range(12):
            self.output += chr(0x20)

        self.output += chr(pattern.thread_count - 1)
        for color in pattern.thread_colors:
            self.output += chr(color)
        for _ in range(0x1cf - pattern.thread_count):
            self.output += chr(0x20)

        self.output += self.write_int(0x0)

        graphics_offset_location = len(self.output)

        self.output += chr(0x00)
        self.output += chr(0x00)
        self.output += chr(0x00)
        self.output += chr(0x31)
        self.output += chr(0xff)
        self.output += chr(0xf0)

        self.output += self.write_int(pattern.bounds.width)
        self.output += self.write_int(pattern.bounds.height)
        self.output += self.write_int(0x1e0)
        self.output += self.write_int(0x1b0)
        self.output += self.write_int(0x9000 | -int(round(pattern.bounds.left)), LE=False)
        self.output += self.write_int(0x9000 | -int(round(pattern.bounds.top)), LE=False)

        thisX = 0.0
        thisY = 0.0
        stopCode = 2
        for block in pattern.blocks:
            for stitch in block.stitches:
                deltaX = int(round(stitch.xx - thisX))
                deltaY = int(round(stitch.yy - thisY))
                thisX += deltaX
                thisY += deltaY

                if "STOP" in stitch.tags or "COLOR" in stitch.tags:
                    self.output += chr(0xFE)
                    self.output += chr(0xB0)
                    self.output += chr(stopCode)
                    if stopCode == 2:
                        stopCode = 1
                    else:
                        stopCode = 2
                elif "END" in stitch.tags:
                    self.output += chr(0xff)
                    break
                elif -64 < deltaX < 63 and -64 < deltaY < 63 and not (
                                "TRIM" in stitch.tags or "JUMP" in stitch.tags):
                    self.output += chr(deltaX + 0x80 if deltaX < 0 else deltaX)
                    self.output += chr(deltaY + 0x80 if deltaY < 0 else deltaY)
                else:
                    self.encode_jump(deltaX, stitch.tags)
                    self.encode_jump(deltaY, stitch.tags)

        graphics_offset_value = len(self.output) - graphics_offset_location + 2
        self.output = self.output[:graphics_offset_location] + \
                      chr(graphics_offset_value & 0xFF) + \
                      chr((graphics_offset_value >> 8) & 0xFF) + \
                      chr((graphics_offset_value >> 16) & 0xFF) + \
                      self.output[graphics_offset_location + 3:]

        populateImage(pattern)
        self.write_image(pattern.image)
        for i in range(pattern.thread_count):
            pattern.image = imageWithFrame
            populateImage(pattern, index=i)
            self.write_image(pattern.image)


def pattern_to_csv(pattern, filename):
    def wrap_string_list(input_list):
        output_list = []
        for i in input_list:
            output_list.append('"'+str(i)+'"')
        return ",".join(output_list)+"\n"
    if isinstance(filename, str) or isinstance(filename, unicode):
        output_file = open(filename, "wb")
    else:
        output_file = filename
    output_file.write(wrap_string_list(
        ["#", "[THREAD_NUMBER]", "[RED]", "[GREEN]", "[BLUE]", "[DESCRIPTION]",
         "[CATALOG_NUMBER]"]))
    for i in range(pattern.thread_count):
        output_file.write(wrap_string_list(["$", str(i + 1), "186", "152", "0", "(null)", "(null)"]))

    output_file.write(wrap_string_list(["#", "[STITCH_TYPE]", "[X]", '[Y]']))
    for block in pattern.blocks:
        for stitch in block.stitches:
            output_file.write(wrap_string_list(["*", stitch.tags[0], stitch.xx, stitch.yy]))
    output_file.close()


def populateImage(pattern, index=None):
    yFactor = 32.0
    xFactor = 42.0
    if pattern.bounds.height > 0:
        yFactor /= pattern.bounds.height
    if pattern.bounds.width > 0:
        xFactor /= pattern.bounds.width

    for i, block in enumerate(pattern.blocks):
        for stitch in block.stitches:
            if index is not None:
                if i != index:
                    continue
            x = int(round((stitch.xx - pattern.bounds.left) * xFactor)) + 3
            y = int(round((stitch.yy - pattern.bounds.top) * yFactor)) + 3
            pattern.image[y][x] = 1


if __name__ == "__main__":
    input_file = 'duck_pattern.csv'
    pattern = csv_to_pattern(input_file)
    bef = BrotherEmbroideryFile(input_file + ".pes")
    bef.write_pattern(pattern)
    # input_file = 'triangle.csv'
