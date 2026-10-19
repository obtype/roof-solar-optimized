import numpy as np

from roofsolar.terrain import plane_from_normal, normal_azimuth_slope, slope_aspect


def test_plane_normal_roundtrip():
    for az in (0, 42, 133, 223, 311):
        for sl in (5, 20, 35, 50):
            a, b = plane_from_normal(az, sl)
            az2, sl2 = normal_azimuth_slope(a, b)
            assert abs(sl2 - sl) < 1e-6
            # azimuth is only defined modulo 360
            assert min(abs(az2 - az), 360 - abs(az2 - az)) < 1e-6


def test_slope_aspect_of_synthetic_plane():
    ps = 0.1
    h = w = 40
    rows, cols = np.mgrid[0:h, 0:w]
    x = cols * ps
    y = -rows * ps
    # a = 0 -> no east-west tilt; b < 0 -> surface rises to the north
    a, b = 0.0, -0.5
    dsm = (a * x + b * y).astype(np.float64)

    slope, aspect = slope_aspect(dsm, ps)
    expect_slope = np.degrees(np.arctan(np.hypot(a, b)))
    expect_az, _ = normal_azimuth_slope(a, b)

    interior = (slice(3, -3), slice(3, -3))
    assert np.allclose(slope[interior], expect_slope, atol=0.5)
    assert np.allclose(aspect[interior], expect_az, atol=0.5)


def test_flat_surface_has_zero_slope():
    dsm = np.full((20, 20), 42.0)
    slope, _ = slope_aspect(dsm, 0.1)
    assert np.allclose(slope, 0.0)
