"""Visualisation helpers - all figures are saved to ``outputs/``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .obstructions import Obstructions
from .panels import Panel
from .segmentation import Facet, facet_polygons

PALETTE = ListedColormap(plt.cm.tab10(np.linspace(0, 1, 10)))


def _px(xy: np.ndarray, pixel_size: float = 0.1):
    """local metres (x east, y north) -> (col, row) for plotting over imshow."""
    return xy[:, 0] / pixel_size, -xy[:, 1] / pixel_size


def _hillshade(dsm: np.ndarray, az: float = 315.0, alt: float = 45.0) -> np.ndarray:
    gy, gx = np.gradient(dsm.astype(np.float64))
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, h = np.radians(az), np.radians(alt)
    return np.sin(h) * np.cos(slope) + np.cos(h) * np.sin(slope) * np.cos(a - aspect + np.pi)


def save_overview(rgb, dsm, slope, aspect, hag, out_dir: Path):
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    ax[0, 0].imshow(rgb); ax[0, 0].set_title("RGB ortho")
    im = ax[0, 1].imshow(dsm, cmap="terrain"); ax[0, 1].set_title("DSM [m]")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.04)
    im = ax[0, 2].imshow(hag, cmap="viridis", vmax=12); ax[0, 2].set_title("height above ground [m]")
    plt.colorbar(im, ax=ax[0, 2], fraction=0.04)
    ax[1, 0].imshow(_hillshade(dsm), cmap="gray"); ax[1, 0].set_title("hillshade")
    im = ax[1, 1].imshow(slope, cmap="magma", vmax=60); ax[1, 1].set_title("slope [deg]")
    plt.colorbar(im, ax=ax[1, 1], fraction=0.04)
    im = ax[1, 2].imshow(aspect, cmap="twilight", vmin=0, vmax=360); ax[1, 2].set_title("aspect [deg]")
    plt.colorbar(im, ax=ax[1, 2], fraction=0.04)
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    p = out_dir / "01_overview.png"; plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    return p


def save_segmentation(rgb, label_map, facets, out_dir: Path, pixel_size: float = 0.1):
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ax[0].imshow(rgb); ax[0].set_title("RGB")
    show = np.ma.masked_where(label_map == 0, (label_map - 1) % 10)
    ax[1].imshow(show, cmap=PALETTE, vmin=0, vmax=9); ax[1].set_title("roof facets (RANSAC + regularised)")
    ax[2].imshow(rgb)
    ax[2].imshow(show, cmap=PALETTE, alpha=0.45, vmin=0, vmax=9)
    for f in facets:
        if f.polygon is not None:
            cx, cy = _px(f.polygon, pixel_size)
            ax[2].plot(np.r_[cx, cx[0]], np.r_[cy, cy[0]], "k-", lw=1)
    ax[2].set_title("facets over RGB")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    p = out_dir / "02_roof_facets.png"; plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    return p


def save_obstructions(rgb, obstructions: Obstructions, mask, out_dir: Path):
    fig, ax = plt.subplots(1, 4, figsize=(26, 7))
    ax[0].imshow(rgb); ax[0].set_title("RGB")
    overlay = rgb.copy()
    overlay[obstructions.chimney] = [255, 0, 0]
    overlay[obstructions.dormer] = [255, 165, 0]
    overlay[obstructions.skylight] = [0, 128, 255]
    ax[1].imshow(overlay); ax[1].set_title("obstructions: chimney / dormer / skylight")
    ax[2].imshow(rgb); ax[2].imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.35)
    ax[2].set_title("target building mask")
    combined = obstructions.combined
    ax[3].imshow(rgb); ax[3].imshow(np.ma.masked_where(~combined, combined), cmap="autumn")
    ax[3].set_title("obstruction union")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    p = out_dir / "03_obstructions.png"; plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    return p


def save_layout_2d(rgb, label_map, facets, panels, obstructions, out_dir: Path, pixel_size: float = 0.1):
    fig, ax = plt.subplots(1, 2, figsize=(22, 9))
    show = np.ma.masked_where(label_map == 0, (label_map - 1) % 10)
    ax[0].imshow(rgb); ax[0].imshow(show, cmap=PALETTE, alpha=0.35, vmin=0, vmax=9)
    for p in panels:
        cx, cy = _px(p.corners_xyz, pixel_size)
        ax[0].plot(np.r_[cx, cx[0]], np.r_[cy, cy[0]], "-", color="#0b3d91", lw=1)
        ax[0].fill(cx, cy, color="#1a73e8", alpha=0.55)
    ax[0].set_title(f"solar layout on RGB ({len(panels)} panels)")
    ax[1].imshow(rgb)
    overlay = obstructions.combined
    ax[1].imshow(np.ma.masked_where(~overlay, overlay), cmap="autumn", alpha=0.6)
    for p in panels:
        cx, cy = _px(p.corners_xyz, pixel_size)
        ax[1].plot(np.r_[cx, cx[0]], np.r_[cy, cy[0]], "-", color="k", lw=0.8)
        ax[1].fill(cx, cy, color="#1a73e8", alpha=0.4)
    ax[1].set_title("panels + obstruction avoidance")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    p = out_dir / "05_panel_layout_2d.png"; plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    return p


def save_roof_3d(dsm, label_map, mask, facets, panels, out_dir: Path, name: str,
                 pixel_size: float = 0.1, step: int = 2, title: str = "",
                 views=((40, -135), (40, 45))):
    """Render the reconstructed roof as a triangulated 3-D surface.

    The surface is sampled from the piecewise-planar facet model. Flush-mounted
    panels are drawn on top. Two complementary viewpoints are rendered so that
    panels on opposite slopes are both visible.
    """
    import matplotlib.tri as mtri

    sub = (slice(None, None, step), slice(None, None, step))
    m = mask[sub] & (label_map[sub] > 0)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    x = xs.astype(float) * pixel_size * step
    y = -ys.astype(float) * pixel_size * step
    z = dsm[sub][ys, xs].astype(float)
    z0 = float(z.min())

    tri = mtri.Triangulation(x, y)
    pad = 0.08 * max(np.ptp(x), np.ptp(y))
    zr = float(z.max() - z0) + 1e-6

    fig = plt.figure(figsize=(22, 10))
    for k, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
        ax.plot_trisurf(x, y, z - z0, triangles=tri.triangles, color="#c9c9c9",
                        linewidth=0, antialiased=False, shade=True)
        for p in panels:
            pc = p.corners_xyz.copy()
            cx, cy = pc.mean(0)[0], pc.mean(0)[1]
            r = int(round(-cy / pixel_size))
            c = int(round(cx / pixel_size))
            zoff = 0.0
            if 0 <= r < dsm.shape[0] and 0 <= c < dsm.shape[1]:
                zoff = float(dsm[r, c]) - float(pc.mean(0)[2])
            pc[:, 2] += zoff - z0 + 0.45
            ax.add_collection3d(Poly3DCollection([pc], facecolor="#1a73e8",
                                                 edgecolor="#ffffff", linewidths=0.8, alpha=1.0))
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.set_ylim(y.min() - pad, y.max() + pad)
        ax.set_zlim(0, zr * 1.3)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), max(1.8 * zr, 0.5 * np.ptp(x))))
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("height [m]")
        ax.view_init(elev=elev, azim=azim)
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    p = out_dir / name
    plt.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


def save_missing_data(before, filled_ref, filled_interp, mask_missing, out_dir: Path):
    fig, ax = plt.subplots(1, 4, figsize=(24, 7))
    ax[0].imshow(before, cmap="terrain")
    ax[0].imshow(np.ma.masked_where(~mask_missing, mask_missing.astype(float)),
                 cmap="Reds", alpha=0.85)
    ax[0].set_title("clipped DSM: missing (nodata) cells")
    ax[1].imshow(before, cmap="terrain"); ax[1].set_title("clipped DSM (raw)")
    ax[2].imshow(filled_ref, cmap="terrain"); ax[2].set_title("filled (reference + inpainting)")
    ax[3].imshow(filled_interp, cmap="terrain"); ax[3].set_title("filled (interpolation only)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    p = out_dir / "07_missing_data.png"; plt.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    return p
