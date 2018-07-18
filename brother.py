import struct
from os.path import basename, splitext, getsize, join

from os import statvfs
from shutil import copyfile

from pattern_utils import csv_to_pattern
from thread import pecThreads

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


class BoundingBox:
    # this is only used when calculating the conversion from pattern to PES files
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


def calc_bounding_box(blocks):
    bounding_rect = BoundingBox()
    for block in blocks:
        stitches = block.stitches
        for stitch in stitches:
            if "TRIM" in stitch.tags:
                continue
            bounding_rect.left = min(bounding_rect.left, stitch.x*10.0)
            bounding_rect.top = min(bounding_rect.top, stitch.y)
            bounding_rect.right = max(bounding_rect.right, stitch.x)
            bounding_rect.bottom = max(bounding_rect.bottom, stitch.y)
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
        self.output += "#PES0001"

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
        pec_location = len(self.output)

        self.output = self.output[:0x08] + \
                      self.write_int(pec_location, width=4) + self.output[
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
        self.output += self.write_int(stitch.x - bounds.left)
        self.output += self.write_int(stitch.y + bounds.top)

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
        stope_code = 2
        for block in pattern.blocks:
            for stitch in block.stitches:
                deltaX = int(round(stitch.x - thisX))
                deltaY = int(round(stitch.y - thisY))
                thisX += deltaX
                thisY += deltaY

                if "STOP" in stitch.tags or "COLOR" in stitch.tags:
                    self.output += chr(0xFE)
                    self.output += chr(0xB0)
                    self.output += chr(stope_code)
                    if stope_code == 2:
                        stope_code = 1
                    else:
                        stope_code = 2
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

        populate_image(pattern)
        self.write_image(pattern.image)
        for i in range(pattern.thread_count):
            pattern.image = imageWithFrame
            populate_image(pattern, index=i)
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
    for i, color in enumerate(pattern.thread_colors):
        output_file.write(wrap_string_list(["$", str(i + 1)]+pecThreads[color][0]+["(null)", "(null)"]))

    output_file.write(wrap_string_list(["#", "[STITCH_TYPE]", "[X]", '[Y]']))
    for block in pattern.blocks:

        for stitch in block.stitches:
            output_file.write(wrap_string_list(["*", stitch.tags[0], stitch.x/10.0, 0.1*(pattern.bounds.height-stitch.y)]))
    output_file.close()


def populate_image(pattern, index=None):
    # this is Brother sewing machine specific
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
            x = int(round((stitch.x - pattern.bounds.left) * xFactor)) + 3
            y = int(round((stitch.y - pattern.bounds.top) * yFactor)) + 3
            pattern.image[y][x] = 1


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
    input_file = 'text.svg.csv'
    pattern = csv_to_pattern(input_file)
    bef = BrotherEmbroideryFile(input_file + ".pes")
    bef.write_pattern(pattern)
    # input_file = 'triangle.csv'
