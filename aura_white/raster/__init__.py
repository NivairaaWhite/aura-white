"""Rasterisation used by the texture stage. The default backend is pure PyTorch (no compiler, works on
CPU / CUDA / MPS on any Python that has PyTorch); NVIDIA's nvdiffrast is used automatically when it is
installed *and* passes a runtime self-test against the PyTorch implementation."""
from .camera import (Camera, fov_to_f, look_at, normalize_mesh, orbit_camera, zero123pp_cameras, transform_camera,
                     ZERO123PP_AZIMUTHS, ZERO123PP_ELEVATIONS, ZERO123PP_FOV, ZERO123PP_RADIUS)
from .core import Raster, RasterOut, get_raster, interpolate

__all__ = ["Camera", "transform_camera", "fov_to_f", "look_at", "normalize_mesh", "orbit_camera", "zero123pp_cameras",
           "ZERO123PP_AZIMUTHS", "ZERO123PP_ELEVATIONS", "ZERO123PP_FOV", "ZERO123PP_RADIUS",
           "Raster", "RasterOut", "get_raster", "interpolate"]
