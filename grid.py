class Grid:
    def __init__(self, paths):
        self.paths = paths
        self.initialize()

    def initialize(self):
        current_grid = defaultdict(dict)
        # simplify paths to lines
        poly_paths = []
        for path in paths:
            if path.length() > minimum_stitch:
                num_segments = ceil(path.length() / minimum_stitch)
                for seg_i in range(int(num_segments)):
                    poly_paths.append(Line(start=path.point(seg_i/num_segments), end=path.point((seg_i+1)/num_segments)))
            else:
                poly_paths.append(Line(start=path.start, end=path.end))
        bbox = overall_bbox(paths)
        curr_x = int(bbox[0]/minimum_stitch)*minimum_stitch
        total_tests = int(bbox[1]-bbox[0])*int(bbox[3]-bbox[2])/(minimum_stitch*minimum_stitch)
        while curr_x < bbox[1]:
            curr_y = int(bbox[2]/minimum_stitch)*minimum_stitch

            while curr_y < bbox[3]:
                test_line = Line(start=curr_x + curr_y * 1j,
                                 end=curr_x + minimum_stitch + (
                                                               curr_y + minimum_stitch) * 1j)
                start = time()
                is_contained = path1_is_contained_in_path2(test_line, Path(*poly_paths))
                end = time()
                if is_contained:
                    current_grid[curr_x][curr_y] = False
                curr_y += minimum_stitch
            curr_x += minimum_stitch
        self.current_grid = current_grid

    def grid_available(self, pos):
        if pos.real in self.current_grid:
            if pos.imag in self.current_grid[pos.real]:
                return not self.current_grid[pos.real][pos.imag]
            else:
                return False
        else:
            return False

    def count_empty(self):
        count = 0
        for x in self.current_grid:
            for y in self.current_grid[x]:
                count += not self.current_grid[x][y]
        return count

    def find_upper_corner(self):
        # find the top or bottom left corner of the grid
        curr_pos = None
        for x in self.current_grid:
            for y in self.current_grid[x]:
                if curr_pos is None:
                    curr_pos = x + 1j * y
                    continue
                if x < curr_pos.real and y < curr_pos.imag and not self.current_grid[x][y]:
                    curr_pos = x + 1j * y
        return curr_pos