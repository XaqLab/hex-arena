import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from typing import Optional
from collections.abc import Collection

from . import rcParams
from .alias import Axes, Artist


class Arena:
    r"""Class for the arena where the monkey forages for food.

    The arena is hexagonal shape, with one of corners at coordinate (1, 0).
    Multiple anchors will be defined to discretize the space. Anchors form a
    hexagonal grid, and the number of intervals along one wall is defined as the
    resolution of anchors.

    """

    def __init__(self,
        resol: Optional[int] = None,
    ):
        r"""
        Args
        ----
        resol:
            Resolution of the hexagonal anchor grid. `resol=1` means only seven
            anchors: six at the corner and one at the center.

        """
        _rcParams = rcParams.get('arena.Arena._init_', {})
        self.resol: int = _rcParams.resol if resol is None else resol
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
                x0, y0 = r*np.cos(theta), r*np.sin(theta)
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
                    x = x0+k/self.resol*np.cos(theta)
                    y = y0+k/self.resol*np.sin(theta)
                    anchors.append((x, y))
        self.anchors: tuple[tuple[float, float]] = tuple(anchors)

    def plot_mesh(self,
        ax: Axes,
    ) -> None:
        _anchors = np.array(self.anchors)
        _idxs = self.corners+[self.corners[0]]
        ax.plot(
            _anchors[_idxs, 0], _anchors[_idxs, 1],
            color='darkgray', linewidth=3, zorder=0
        )
        ax.scatter(
            _anchors[:, 0], _anchors[:, 1],
            s=40, marker='X', color='yellow',
        )
        ax.scatter(
            _anchors[self.boxes, 0], _anchors[self.boxes, 1], s=120,
            marker='o', facecolors='none', edgecolors='red', linewidths=2,
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
            ax.plot(segment[0], segment[1], color='green', linestyle='--', linewidth=1)

        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect('equal')
        ax.set_axis_off()

    def plot_tile(self,
        ax: Axes,
        tile_idx: Optional[int], color = 'none',
        h_tile: Optional[Artist] = None,
    ) -> Artist:
        if tile_idx is None:
            xy = np.full((1, 2), fill_value=np.nan)
        else:
            xy = np.stack([
                np.array([np.cos(theta), np.sin(theta)])/(2*self.resol)
                for theta in [i/3*np.pi+np.pi/6 for i in range(6)]
            ])+self.anchors[tile_idx]
        if h_tile is None:
            h_tile = ax.add_patch(Polygon(
                xy, edgecolor='none', facecolor=color, zorder=-1,
            ))
        else:
            h_tile.set_xy(xy)
            h_tile.set_facecolor(color)
        return h_tile

    def plot_heatmap(self,
        heatmap: Collection[float],
        figsize: tuple[float, float] = None,
        cmap: str = 'YlOrBr',
        vmin: float = 0,
        vmax: Optional[float] = None,
        clabel: str = '',
    ):
        r"""Plots heat map over the arena.

        Args
        ----
        heatmap: (num_tiles,)
            An array containing non-negative values for each tile.
        figsize:
            Figure size.
        cmap:
            Color map string.
        vmin, vmax:
            Min and max value for color map.
        clabel:
            Color bar label.

        Returns
        -------
        fig, ax:
            Figure and axis handle.

        """
        assert len(heatmap)==self.num_tiles
        assert np.all(np.array(heatmap)>=0)
        if figsize is None:
            figsize = (4.5, 4)
        cmap = plt.get_cmap(cmap)
        if vmax is None:
            vmax = np.array(heatmap).max()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.05, 0.8, 0.9])
        self.plot_mesh(ax)

        _xy = np.stack([
            np.array([np.cos(theta), np.sin(theta)])/(2*self.resol)
            for theta in [i/3*np.pi+np.pi/6 for i in range(6)]
        ])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        for i in range(self.num_tiles):
            xy = _xy+self.anchors[i]
            ax.add_patch(Polygon(
                xy, edgecolor='none', facecolor=cmap(norm(heatmap[i])), zorder=-1,
            ))
        plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax, fraction=0.1, shrink=0.8, pad=0.1,
            orientation='horizontal', label=clabel,
        )
        return fig, ax

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
