class UbongoCard(object):

    def __init__(self):
        self.image = None
        self.contour = None
        self.playFieldPattern = None
        self.playFieldContour = []
        self.blocks = []
        self.blockPositions = []
        self.isValid = False

    def __str__(self):
        desc = f"Number of Blocks {len(self.blocks)}"
        return desc


