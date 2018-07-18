from brother import imageWithFrame, calc_bounding_box


class Pattern():
    def __init__(self):
        self.blocks = []
        self.stitches = []
        self.colors = []
        self.current_color = 0
        # this is Brother sewing machine specific - generates the image that appears in
        # the LCD screen
        self.image = imageWithFrame
        self.bounding_box = None

    def add_block(self, block):
        # does some validation on adding a block
        if len(block.stitches) == 0:
            raise ValueError("adding block with no stitches! %s" % block)
        self.blocks.append(block)

    @property
    def all_stitches(self):
        _all_stitches = []
        for block in self.blocks:
            _all_stitches += list(block)
        return _all_stitches

    @property
    def bounds(self):
        if self.bounding_box is None:
            self.bounding_box = calc_bounding_box(self.blocks)
        return self.bounding_box

    @property
    def thread_count(self):
        # go through the list of blocks, tally up the number of times the thread color
        # changes
        return len([i for i in range(1, len(self.blocks))
                    if self.blocks[i - 1].color != self.blocks[i].color]) + 1

    @property
    def thread_colors(self):
        blocks_with_stitches = [block for block in self.blocks if len(block.stitches) > 0]
        if len(blocks_with_stitches) != len(self.blocks) or len(blocks_with_stitches) == 0:
            raise ValueError("This pattern has a block with no stitches!")
        # return set([block.color for block in self.blocks])
        # look for trim, or end not color, to signify the end of the pattern
        return [blocks_with_stitches[0].color]+[block.color for block in blocks_with_stitches
                if "COLOR" in block.stitches[0].tags]

    def __str__(self):
        output = ""
        for i, block in enumerate(self.blocks):
            output += "Block Number: %s, color %s, number of stitches: %s, type: %s\n" % \
                      (i, block.color, block.num_stitches, block.stitches[-1].tags[0])
        return output

    def __repr__(self):
        return self.__str__()
