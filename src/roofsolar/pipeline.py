"""End-to-end pipeline: DSM + RGB -> reconstructed roof + solar-panel layout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from . import buildings, io, obstructions as obst, panels as panels_mod, preprocess, reconstruction, segmentation, terrain, viz
from .coords import px_to_xy
from .logging_utils import get_logger
from .solar import annual_factor, sun_vectors

log = get_logger("pipeline")


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def _site_latlon(grid):
    from rasterio.warp import transform as warp_transform
    x = grid.transform.c + grid.width * 0.5 * grid.transform.a
    y = grid.transform.f + grid.height * 0.5 * grid.transform.e
    lon, lat = warp_transform(grid.crs, "EPSG:4326", [x], [y])
    return float(lat[0]), float(lon[0])


def run(cfg: dict) -> dict:
    data_dir = Path(cfg["data_dir"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ps = float(cfg.get("pixel_size", 0.1))

    # ---- load -------------------------------------------------------------
    serie = io.load_scene(data_dir)
    dsm_raw = serie["dsm"]
    rgb = serie["rgb"]
    grid = serie["dsm_grid"]
    lat, lon = _site_latlon(grid)
    log.info("site latitude=%.4f longitude=%.4f", lat, lon)

    # ---- missing-data demonstration (clipped DSM) -------------------------
    roi = io.clip_roi_mask(serie)
    clip_full = np.full(dsm_raw.shape, np.nan, np.float32)
    clip = serie["dsm_clipped"]
    cnd = serie["dsm_clipped_nodata"]
    valid = (clip != cnd) if cnd is not None else np.isfinite(clip)
    cgrid = serie["dsm_clipped_grid"]
    r_off = int(round((grid.transform.f - cgrid.transform.f) / abs(grid.transform.e)))
    c_off = int(round((cgrid.transform.c - grid.transform.c) / abs(grid.transform.a)))
    h = min(clip.shape[0], dsm_raw.shape[0] - r_off)
    w = min(clip.shape[1], dsm_raw.shape[1] - c_off)
    sub = np.full((h, w), np.nan, np.float32)
    sub[valid[:h, :w]] = clip[:h, :w][valid[:h, :w]]
    clip_full[r_off:r_off + h, c_off:c_off + w] = sub
    clip_box = np.zeros(dsm_raw.shape, bool)
    clip_box[r_off:r_off + h, c_off:c_off + w] = True
    missing = clip_box & ~np.isfinite(clip_full)
    filled_ref = io.fill_nodata(clip_full, np.nan, reference=dsm_raw)
    filled_interp = io.fill_nodata(clip_full, np.nan, reference=None)
    viz.save_missing_data(np.where(np.isfinite(clip_full), clip_full, np.nan),
                          filled_ref, filled_interp, missing, out_dir)
    log.info("clipped DSM missing fraction within its extent: %.1f%% (%.0f cells)",
             100.0 * missing.sum() / max(clip_box.sum(), 1), missing.sum())

    # ---- preprocess -------------------------------------------------------
    dsm = preprocess.denoise(dsm_raw, cfg["preprocess"]["denoise_sigma"])
    roughness = preprocess.local_roughness(dsm, cfg["preprocess"]["roughness_window"])
    ground = preprocess.estimate_ground(dsm, cfg["ground"]["downsample"],
                                        cfg["ground"]["window"], cfg["ground"]["percentile"])
    hag = preprocess.above_ground(dsm, ground)
    veg = preprocess.vegetation_mask(rgb)
    slope, aspect = terrain.slope_aspect(dsm, ps)

    viz.save_overview(rgb, dsm, slope, aspect, hag, out_dir)

    # ---- buildings --------------------------------------------------------
    bres = buildings.run(hag, roughness, veg, roi, **cfg["buildings"])
    target = bres.target_mask

    # ---- roof facet segmentation -----------------------------------------
    facets = segmentation.segment(dsm, target, ps, **cfg["segmentation"])
    label_map = segmentation.facets_label_map(facets, dsm.shape, target, dsm, ps, thresh=None)
    segmentation.facet_polygons(label_map, facets, ps)
    viz.save_segmentation(rgb, label_map, facets, out_dir)

    # ---- obstructions -----------------------------------------------------
    ob = obst.detect(rgb, dsm, facets, target, ps, **cfg["obstructions"])
    viz.save_obstructions(rgb, ob, target, out_dir)

    # ---- 3-D reconstruction ----------------------------------------------
    step = int(cfg.get("mesh_step", 3))
    verts, faces = reconstruction.build_roof_mesh(facets, label_map, target, ps, step=step)
    reconstruction.export_obj(out_dir / "roof_mesh.obj", verts, faces)
    viz.save_roof_3d(dsm, label_map, target, facets, [], out_dir, "04_3d_roof.png", ps, step,
                     "reconstructed roof surface (measured DSM, denoised)")

    # ---- solar layout -----------------------------------------------------
    suns = sun_vectors(lat)
    pcfg = {k: v for k, v in cfg["panels"].items() if k != "wp_per_panel"}
    layout = panels_mod.plan(rgb, dsm, facets, target, ob, ps, suns, **pcfg)
    viz.save_layout_2d(rgb, label_map, facets, layout.panels, ob, out_dir)
    viz.save_roof_3d(dsm, label_map, target, facets, layout.panels, out_dir, "06_panel_layout_3d.png",
                     ps, step, f"reconstructed roof + {len(layout.panels)} flush-mounted panels")
    reconstruction.export_panel_obj(out_dir / "panels.obj", layout.panels)

    # ---- exports ----------------------------------------------------------
    _export_panels_geojson(out_dir / "panels.geojson", layout.panels, grid)
    _export_facets_geojson(out_dir / "facets.geojson", facets, grid)
    _export_interactive(out_dir / "roof_and_panels.html", verts, faces, layout.panels)

    n = len(layout.panels)
    wp = cfg["panels"].get("wp_per_panel", 430)
    metrics = {
        "site": {"latitude": lat, "longitude": lon, "crs": str(grid.crs)},
        "pixel_size_m": ps,
        "panels": {
            "count": n,
            "kwp": round(n * wp / 1000.0, 2),
            "module_w": wp,
            "module_size_m": [cfg["panels"]["panel_w"], cfg["panels"]["panel_h"]],
            "total_area_m2": round(sum(p.panel_w * p.panel_h for p in layout.panels), 2),
        },
        "facets": layout.facet_stats,
        "skipped_facets": layout.skipped_facets,
        "obstructions": ob.counts(),
    }
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("panels=%d  kWp=%.2f  facets=%d  obstructions=%s", n, metrics["panels"]["kwp"],
             len(layout.facet_stats), metrics["obstructions"])
    return metrics


def _export_panels_geojson(path: Path, panels, grid):
    feats = []
    for i, p in enumerate(panels):
        ring = [[float(x + grid.transform.c), float(y + grid.transform.f), float(z)]
                for x, y, z in np.vstack([p.corners_xyz, p.corners_xyz[0]])]
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "properties": {"id": i, "facet": p.facet_id,
                                     "tilt_deg": round(p.tilt_deg, 1),
                                     "azimuth_deg": round(p.azimuth_deg, 1),
                                     "irradiance": round(p.irradiance, 3)}})
    _write_geojson(path, feats, grid)


def _export_facets_geojson(path: Path, facets, grid):
    feats = []
    for f in facets:
        if f.polygon is None:
            continue
        ring = [[float(x + grid.transform.c), float(y + grid.transform.f), float(z)]
                for x, y, z in np.vstack([f.polygon, f.polygon[0]])]
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": [ring]},
                      "properties": {"id": f.id, "slope_deg": round(f.slope, 2),
                                     "azimuth_deg": round(f.azimuth, 2),
                                     "area_m2": round(f.area_m2, 2), "rms_m": round(f.rms, 3)}})
    _write_geojson(path, feats, grid)


def _write_geojson(path: Path, features, grid):
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": str(grid.crs)}},
          "features": features}
    with open(path, "w") as fh:
        json.dump(fc, fh, indent=1)


def _export_interactive(path: Path, verts, faces, panels):
    try:
        import plotly.graph_objects as go
    except Exception:  # pragma: no cover
        return
    fig = go.Figure()
    if len(faces):
        fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                color="lightgray", opacity=0.9, name="roof"))
    for idx, p in enumerate(panels):
        c = p.corners_xyz
        xx = list(c[:, 0]) + [c[0, 0]]
        yy = list(c[:, 1]) + [c[0, 1]]
        zz = list(c[:, 2]) + [c[0, 2]]
        fig.add_trace(go.Scatter3d(x=xx, y=yy, z=zz, mode="lines",
                                   line=dict(color="royalblue", width=4),
                                   showlegend=idx == 0, name="panels"))
    fig.update_layout(scene_aspectmode="data", title="Reconstructed roof and solar panels")
    fig.write_html(str(path), include_plotlyjs="cdn")
