class Stitch:
    def __init__(self, tags=None, xx=0.0, yy=0.0, color=0):
        if tags is None:
            tags = []
        self.tags = tags
        self.x = xx
        self.y = yy
        self.color = color

    @property
    def complex(self):
        return self.x + self.y*1j

    def __str__(self):
        return "{} {} {}\n".format(self.x, self.y, self.tags[0])

    def __repr__(self):
        return self.__str__()