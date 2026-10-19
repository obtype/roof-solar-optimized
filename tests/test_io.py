import numpy as np

from roofsolar.io import fill_nodata


def test_fill_nodata_with_reference():
    arr = np.full((30, 30), 10.0, np.float32)
    arr[10:20, 10:20] = -9999.0
    ref = np.full((30, 30), 12.0, np.float32)
    out = fill_nodata(arr, -9999.0, reference=ref)
    assert not np.any(out == -9999.0)
    # the hole is taken from the reference
    assert np.allclose(out[10:20, 10:20], 12.0)


def test_fill_nodata_interpolation_only():
    arr = np.full((30, 30), 5.0, np.float32)
    arr[14:16, 14:16] = np.nan
    out = fill_nodata(arr, np.nan, reference=None)
    assert np.all(np.isfinite(out))
    # surrounded by 5.0 -> the interpolation converges to 5.0
    assert np.allclose(out[14:16, 14:16], 5.0, atol=1e-3)
