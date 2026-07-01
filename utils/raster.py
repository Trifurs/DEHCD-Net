from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


def read_raster(path: str | Path) -> Tuple["Any", Dict[str, Any]]:
    """Read raster-like imagery as CHW array and return lightweight metadata."""
    path = Path(path)
    rasterio_error = None
    try:
        import rasterio

        with rasterio.open(path) as src:
            array = src.read()
            meta = {
                "driver": src.driver,
                "count": src.count,
                "height": src.height,
                "width": src.width,
                "dtype": str(array.dtype),
                "crs": str(src.crs) if src.crs else None,
                "transform": tuple(src.transform) if src.transform else None,
                "resolution": tuple(src.res) if src.res else None,
                "nodata": src.nodata,
            }
            return array, meta
    except ImportError as exc:
        rasterio_error = exc
    except Exception as exc:
        rasterio_error = exc

    try:
        import numpy as np
        from PIL import Image

        image = Image.open(path)
        array = np.asarray(image)
        if array.ndim == 2:
            array = array[None, ...]
        else:
            array = array.transpose(2, 0, 1)
        meta = {
            "driver": "PIL",
            "count": int(array.shape[0]),
            "height": int(array.shape[1]),
            "width": int(array.shape[2]),
            "dtype": str(array.dtype),
            "crs": None,
            "transform": None,
            "resolution": None,
            "nodata": None,
        }
        return array, meta
    except ImportError as exc:
        raise ImportError(
            "Could not read image. Install rasterio for GeoTIFF or Pillow+numpy for common images."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Could not read {path}: rasterio={rasterio_error}; pillow={exc}") from exc
