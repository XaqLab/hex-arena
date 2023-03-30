class Arena:
    r"""Class for the arena where the monkey forages for food.

    The arena is hexagonal shape, with one of corners at coordinate (0, 1).
    Multiple anchors will be defined to discretize the space. Anchors form a
    hexagonal grid, and the number of intervals along one wall is defined as the
    resolution of anchors.

    """

    def __init__(self,
        resol: int,
    ):
        r"""
        Args
        ----
        resol:
            Resolution of the hexagonal anchor grid. `resol=1` means only seven
            anchors: six at the corner and one at the center.

        """
        self.resol = resol
