"""Texturing engine: UV unwrap, decimation, camera registration, multi-view baking, normal baking."""
from .bake import BakeResult, ViewSource, bake_texture
from .decimate import decimate
from .fill import dilate_gutters, fill_from_neighbors
from .normals import bake_normal_map, tangent_frames
from .register import (Registration, align_view, estimate_foreground, find_orientation, make_camera,
                       orientation_from_views, refine_reference_photometric,
                       register_reference)
from .unwrap import unwrap

__all__ = ["BakeResult", "ViewSource", "bake_texture", "decimate", "dilate_gutters", "fill_from_neighbors",
           "bake_normal_map", "tangent_frames", "Registration", "align_view", "estimate_foreground",
           "find_orientation", "make_camera", "orientation_from_views", "refine_reference_photometric", "register_reference", "unwrap"]
