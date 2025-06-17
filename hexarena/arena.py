import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from jarvis.config import Config

from .alias import Array, Axes, Artist


class Arena:
    r"""Class for the arena where the monkey forages for food.

    The arena is hexagonal shape, with one of corners at coordinate (1, 0).
    Multiple anchors will be defined to discretize the space. Anchors form a
    hexagonal grid, and the number of intervals along one wall is defined as the
    resolution of anchors.

    Args
    ----
    resol:
        Resolution of the hexagonal anchor grid. `resol=1` means only seven
        anchors: six at the corner and one at the center.

    """

    def __init__(self,
        resol: int = 2,
    ):
        self.resol = resol
        assert self.resol%2==0, (
            "'resol' needs to be an even number to allocate anchors on the center of walls."
        )

        self.num_boxes = 3
        self.num_tiles = 3*self.resol**2+3*self.resol+1
        anchors = []
        self.corners: list[int] = []
        self.boxes: list[int] = []
        self.inners: list[int] = []
        self.outers: list[int] = []
        for i in range(self.resol+1):
            if i==0:
                self.center = 0
                anchors.append((0., 0.))
                continue
            r = i/self.resol
            for j in range(6):
                theta = j/6*(2*np.pi)
                x0, y0 = r*np.cos(theta).item(), r*np.sin(theta).item()
                theta += 2*np.pi/3
                for k in range(i):
                    t_idx = len(anchors)
                    if i==self.resol:
                        self.outers.append(t_idx)
                        if k==0:
                            self.corners.append(t_idx)
                        elif j%2==0 and k==self.resol//2:
                            self.boxes.append(t_idx)
                    else:
                        self.inners.append(t_idx)
                    x = x0+k/self.resol*np.cos(theta).item()
                    y = y0+k/self.resol*np.sin(theta).item()
                    anchors.append((x, y))
        self.anchors: tuple[tuple[float, float]] = tuple(anchors)

    def __repr__(self) -> str:
        return f"Arena of size {self.resol} with {self.num_boxes} boxes"

    @property
    def spec(self) -> dict:
        return {
            '_target_': 'hexarena.arena.Arena',
            'resol': self.resol,
        }

    def plot_mesh(self,
        ax: Axes,
    ) -> None:
        r"""Plots arena mesh.

        Args
        ----
        ax:
            Axis to plot on.

        """
        _anchors = np.array(self.anchors)
        _idxs = self.corners+[self.corners[0]]
        ax.plot(
            _anchors[_idxs, 0], _anchors[_idxs, 1],
            color='darkgray', linewidth=3, zorder=0
        )
        ax.scatter(
            _anchors[self.boxes, 0], _anchors[self.boxes, 1], s=120,
            marker='o', facecolors='none', edgecolors='lime', linewidths=2,
        )

        def append_seg(x, y, theta):
            segments.append(np.array([
                x+s_len/3**0.5*np.cos([theta-np.pi/6, theta+np.pi/6]),
                y+s_len/3**0.5*np.sin([theta-np.pi/6, theta+np.pi/6]),
            ]))
        s_len, segments = 1/self.resol, []
        for j in range(6):
            theta = j/6*(2*np.pi)
            for i in range(self.resol+1):
                if i==0:
                    append_seg(0, 0, theta)
                else:
                    r = i/self.resol
                    x0, y0 = r*np.cos(theta), r*np.sin(theta)
                    for k in range(i):
                        x = x0+k/self.resol*np.cos(theta+2*np.pi/3)
                        y = y0+k/self.resol*np.sin(theta+2*np.pi/3)
                        for l in range(-1 if k==0 else 0, 3):
                            append_seg(x, y, theta+l*np.pi/3)
        for segment in segments:
            ax.plot(segment[0], segment[1], color='silver', linestyle='--', linewidth=1/self.resol)

        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect('equal')
        ax.set_axis_off()

    def plot_tile(self,
        ax: Axes,
        tile_idx: int, color = 'none', size = 2/3**0.5,
        h_tile: Artist|None = None,
    ) -> Artist:
        r"""Colors one tile.

        Args
        ----
        ax:
            Axis to plot on.
        tile_idx:
            Index of tile to color, in [0, num_tiles).
        color:
            The desired tile color.
        size:
            Size of the patch relative to the tile
        h_tile:
            Handle of an existing tile. If is ``None``, create a new one.
            Otherwise update the color and position of provided handle.

        Returns
        -------
        h_tile:
            Handle of the tile patch.

        """
        xy = np.stack([
            np.array([np.cos(theta), np.sin(theta)])/(2*self.resol)*size
            for theta in [i/3*np.pi+np.pi/6 for i in range(6)]
        ])+(self.anchors[tile_idx] if tile_idx>=0 else np.full((2,), fill_value=np.nan))
        if h_tile is None:
            h_tile = ax.add_patch(Polygon(
                xy, edgecolor='none', facecolor=color, zorder=-1,
            ))
        else:
            h_tile.set_xy(xy)
            h_tile.set_facecolor(color)
        return h_tile

    def plot_map(self,
        ax: Axes,
        vals: Array,
        door: bool = False,
        cmap: str = 'YlOrBr',
        vmin: float = 0,
        vmax: float|None = None,
        cbar_kw: dict|None = None,
        h_tiles: list[Artist]|None = None,
    ) -> list[Artist]:
        r"""Plots heat map over the arena.

        Args
        ----
        ax:
            Axis to plot on.
        vals: (num_tiles,)
            An array containing non-negative values for each tile.
        cmap, vmin, vmax:
            Color map and the extremum values.
        clabel:
            Color bar label.
        h_tiles:
            Handles of existing tiles. If is ``None``, create new ones.
            Otherwise update colors of provided handles.

        Returns
        -------
        h_tiles:
            Handles of colored tiles.

        """
        cbar_kw = Config(cbar_kw).fill({
            'fraction': 0.1, 'shrink': 0.8, 'pad': 0.05,
            'orientation': 'horizontal', 'location': 'bottom', 'disable': False,
        })
        disable_pbar = cbar_kw.pop('disable')
        vals = np.array(vals)
        assert len(vals)==self.num_tiles
        assert np.all(vals>=0)
        cmap = plt.get_cmap(cmap)
        if vmax is None:
            vmax = vals.max()
        self.plot_mesh(ax)
        if door:
            x = [np.cos(np.pi/3), 1]
            y = [-np.sin(np.pi/3), 0]
            p = 0.35
            ax.plot(
                [(1-p)*x[0]+p*x[1], p*x[0]+(1-p)*x[1]],
                [(1-p)*y[0]+p*y[1], p*y[0]+(1-p)*y[1]],
                color='dimgray', linewidth=5,
            )

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        if h_tiles is None:
            h_tiles = []
            for i in range(self.num_tiles):
                h_tiles.append(self.plot_tile(ax, i, cmap(norm(vals[i]))))
            if not disable_pbar:
                plt.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, **cbar_kw,
                )
        else:
            assert len(h_tiles)==self.num_tiles
            for i in range(self.num_tiles):
                self.plot_tile(ax, i, cmap(norm(vals[i])), h_tile=h_tiles[i])
        return h_tiles

    def is_inside(self,
        xy: tuple[float, float],
    ) -> bool:
        r"""Returns if a position is inside the arena."""
        x, y = xy
        for i in range(6):
            x0, y0 = self.anchors[self.corners[i]]
            x1, y1 = self.anchors[self.corners[(i+1)%6]]
            dx, dy = x-x0, y-y0
            dx1, dy1 = x1-x0, y1-y0
            if dx*dy1-dx1*dy>0:
                return False
        return True

    def nearest_tile(self,
        xy: tuple[float, float],
    ) -> int:
        r"""Get integer index of a nearest tile."""
        d = ((np.array(xy)-np.array(self.anchors))**2).sum(axis=1)**0.5
        s_idx = np.argmin(d)
        return s_idx
