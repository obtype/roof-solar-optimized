import numpy as np
from shapely.geometry import box

from roofsolar.obstructions import Obstructions
from roofsolar.panels import FacetFrame, _pack_polygon, plan
from roofsolar.segmentation import Facet
from roofsolar.terrain import normal_azimuth_slope, plane_from_normal


def _south_facet(c=10.0):
    a, b = plane_from_normal(180.0, 35.0)
    az, sl = normal_azimuth_slope(a, b)
    return Facet(id=1, a=a, b=b, c=c, azimuth=az, slope=sl, area_m2=16.0,
                 rms=0.0, pixel_count=1600, centroid=(2.0, -2.0))


def test_facet_frame_roundtrip():
    f = _south_facet()
    frame = FacetFrame(f)
    pts = np.array([[1.0, -1.0, f.z_at(1.0, -1.0)],
                    [3.0, -2.0, f.z_at(3.0, -2.0)]])
    uv = frame.to_uv(pts)
    back = frame.to_xyz(uv)
    assert np.allclose(back, pts, atol=1e-9)


def test_pack_polygon_respects_bounds():
    usable = box(0.0, 0.0, 5.0, 3.0)
    rects = _pack_polygon(usable, 1.72, 1.13, 0.03)
    assert len(rects) >= 4
    for r in rects:
        assert usable.covers(r)
    # non-overlapping
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not rects[i].intersects(rects[j])


def test_plan_places_flush_panels():
    ps = 0.1
    grid = 60
    f = _south_facet()
    ys, xs = np.mgrid[0:grid, 0:grid]
    x = xs * ps
    y = -ys * ps
    dsm = f.z_at(x, y).astype(np.float32)
    target = np.zeros((grid, grid), bool)
    target[10:50, 10:50] = True
    obst = Obstructions(np.zeros_like(target), np.zeros_like(target), np.zeros_like(target))

    res = plan(np.zeros((grid, grid, 3), np.uint8), dsm, [f], target, obst,
               pixel_size=ps, suns=None, min_slope=10.0, max_slope=60.0)

    assert len(res.panels) >= 1
    for p in res.panels:
        # every corner lies exactly on the facet plane (flush-mounted)
        for cx, cy, cz in p.corners_xyz:
            assert abs(cz - f.z_at(cx, cy)) < 1e-6
        # inside the building mask
        r = int(round(-p.center_xyz[1] / ps))
        c = int(round(p.center_xyz[0] / ps))
        assert target[r, c]
