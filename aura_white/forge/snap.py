"""Silhouette snapping.

After registration a neural mesh matches the artwork silhouette to within a few pixels.  This pulls the outline of the
mesh onto the artwork outline (blade tips, fur tips, fingers) by moving silhouette vertices *within the image plane*
(their depth is kept), then relaxing the displacement over the mesh so the surface stays smooth.  Interior vertices are
moved only by the relaxation, never directly, so the reference view becomes silhouette-exact while the shape seen from
other angles changes as little as possible.  NumPy + SciPy only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import morph, softraster as sr


@dataclass
class SnapResult:
    verts: np.ndarray
    moved: int
    mean_shift_px: float
    iou_before: float
    iou_after: float


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def _laplacian(n: int, faces: np.ndarray):
    from scipy import sparse

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    return A, np.maximum(deg, 1.0)


def snap_to_silhouette(verts: np.ndarray, faces: np.ndarray, cam, ref_alpha: np.ndarray, max_shift_px: float = 0.0,
                       relax: int = 12, depth_tol: float = 0.03, iterations: int = 2) -> SnapResult:
    from scipy import ndimage as ndi

    H, W = ref_alpha.shape
    S = float(max(H, W))
    ref = ref_alpha >= 0.5
    max_shift = max_shift_px or 0.02 * S
    v = np.asarray(verts, np.float64).copy()
    f = np.asarray(faces, np.int64)
    A, deg = _laplacian(len(v), f)
    # signed distance of the artwork silhouette (positive inside) and its gradient
    sd = morph.edt(ref) - morph.edt(~ref)
    sd = ndi.gaussian_filter(sd.astype(np.float32), 1.0)
    gy, gx = np.gradient(sd)
    mask0, depth0, _ = sr.render_camera(cam, v, f, int(H), int(W))
    iou_before = _iou(mask0, ref)
    moved_total = 0
    shift_sum = 0.0
    for _ in range(iterations):
        mask, depth, _t = sr.render_camera(cam, v, f, int(H), int(W))
        xy, w = sr.project(cam, v, int(H), int(W))
        ix = np.clip(np.floor(xy[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.floor(xy[:, 1]).astype(int), 0, H - 1)
        # silhouette vertices: visible (depth agrees with the z-buffer) and within 1.5 px of the mesh outline
        inside_d = morph.edt(mask)
        outside_d = morph.edt(~mask)
        edge_d = np.where(mask, inside_d - 0.5, outside_d - 0.5)
        vis = np.abs(depth[iy, ix] - w) <= depth_tol * np.maximum(w, 1e-6)
        sil = (edge_d[iy, ix] <= 1.5) & vis & (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
        if not sil.any():
            break
        s = ndi.map_coordinates(sd, [xy[sil, 1] - 0.5, xy[sil, 0] - 0.5], order=1, mode="nearest")
        gxs = ndi.map_coordinates(gx, [xy[sil, 1] - 0.5, xy[sil, 0] - 0.5], order=1, mode="nearest")
        gys = ndi.map_coordinates(gy, [xy[sil, 1] - 0.5, xy[sil, 0] - 0.5], order=1, mode="nearest")
        gn = np.maximum(np.hypot(gxs, gys), 1e-6)
        shift = np.clip(-s, -max_shift, max_shift)                 # inside (s>0) -> move outward (negative direction)
        d_px = np.stack([gxs / gn * (-shift) * -1.0, gys / gn * (-shift) * -1.0], 1)
        # gradient points inward; to move by `shift` px towards the boundary: p' = p + n_in * (-s)
        d_px = np.stack([gxs / gn, gys / gn], 1) * (-np.clip(s, -max_shift, max_shift))[:, None]
        target = xy[sil] + d_px
        newp = sr.unproject(cam, target[:, 0], target[:, 1], w[sil], int(H), int(W))
        disp = np.zeros_like(v)
        wgt = np.zeros(len(v))
        disp[sil] = newp - v[sil]
        wgt[sil] = 1.0
        idx = np.flatnonzero(sil)
        # relax: diffuse the displacement over the mesh, keep silhouette vertices pinned to their targets
        d = disp.copy()
        for _r in range(relax):
            nd = (A @ d) / deg[:, None]
            d = np.where(sil[:, None], disp, 0.5 * d + 0.5 * nd)
        v = v + d
        moved_total += int(sil.sum())
        shift_sum += float(np.abs(s).mean())
    mask1, _, _ = sr.render_camera(cam, v, f, int(H), int(W))
    return SnapResult(v, moved_total, shift_sum / max(iterations, 1), iou_before, _iou(mask1, ref))
