from embroidery_thread import nearest_color, pecThreads


class Block:
    # a block is a contiguous set of stitches
    def __init__(self, stitches=None, color=None):
        if stitches is None:
            stitches = []
        if color is None:
            color = [0, 0, 0]
        self.stitches = stitches
        self.color = nearest_color(color)
        if not isinstance(self.color, int):
            raise ValueError(
                "block was instantiated with something that was not a color: %s %s %s" % ( type(self.color),
                    self.color, color))

    @property
    def stitch_type(self):
        return 1 if "JUMP" in self.stitches[0].tags else 0

    @property
    def num_stitches(self):
        return len(self.stitches)  # - self.stitch_type


    @property
    def tuple_color(self):
        return tuple(pecThreads[self.color][0])

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

    def __iter__(self):
        for i in range(len(self.stitches)):
            yield (self.stitches[i].x, self.stitches[i].y)