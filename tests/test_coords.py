import numpy as np

from roofsolar.coords import px_to_xy, xy_to_px


def test_px_to_xy_axes():
    # column grows east (+x), row grows south (-y)
    x, y = px_to_xy([0, 0, 10], [0, 10, 0], pixel_size=0.5)
    assert np.allclose(x, [0.0, 5.0, 0.0])
    assert np.allclose(y, [0.0, 0.0, -5.0])


def test_roundtrip():
    rows = np.array([0, 3, 17, 42])
    cols = np.array([0, 5, 9, 100])
    x, y = px_to_xy(rows, cols, 0.1)
    r, c = xy_to_px(x, y, 0.1)
    assert np.allclose(r, rows)
    assert np.allclose(c, cols)
