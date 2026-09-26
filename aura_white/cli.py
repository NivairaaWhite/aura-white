"""Command line: studio | generate | serve | fetch | convert | setup | doctor.  ASCII-only output (Windows consoles)."""
from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

from . import __version__

COMMANDS = ("studio", "generate", "serve", "fetch", "convert", "setup", "doctor", "forge", "repair", "eval",
            "shape-train", "shape-complete")


def _progress_printer():
    tty = sys.stdout.isatty()
    last = {"msg": None}

    def cb(msg: str, frac: float):
        if tty:
            sys.stdout.write(f"\r  {msg:<48} {frac * 100:5.1f}%")
            sys.stdout.flush()
            if frac >= 1.0:
                sys.stdout.write("\n")
        elif msg != last["msg"]:
            print(f"  {msg}", flush=True)
        last["msg"] = msg

    return cb


def _load_engine(a, progress=None):
    from .pipeline import AuraWhite

    return AuraWhite.from_pretrained(a.weights, a.device, a.precision, trust_pickle=a.trust_checkpoint, progress=progress)


def cmd_generate(a) -> int:
    from .pipeline import EmptyMeshError

    engine = _load_engine(a, _progress_printer())
    info = engine.info()
    print(f"model: {info['parameters_m']} M params on {info['device']} ({info['dtype']})")
    out = Path(a.out)
    fails = 0
    for img in a.images:
        p = Path(img)
        if not p.is_file():
            print(f"skip: {img} (not found)")
            fails += 1
            continue
        print(f"\n{p.name}")
        try:
            res = engine.generate(p, resolution=a.resolution, threshold=a.threshold, remove_bg=not a.no_remove_bg,
                                  foreground_ratio=a.foreground_ratio, matting=a.matting, adaptive=not a.dense,
                                  smooth=a.smooth, min_part_ratio=a.min_part_ratio, progress=_progress_printer())
        except EmptyMeshError as e:
            print(f"  failed: {e}")
            fails += 1
            continue
        folder = out / p.stem
        files = res.save_all(folder, "mesh", tuple(a.format), up=a.up, yaw=a.yaw)
        res.prepared.preview.save(folder / "input_processed.png")
        if a.turntable:
            print("  rendering turntable (slow on CPU) ...")
            engine.turntable(res, folder / "turntable.gif")
        s, t = res.stats, res.timings
        print(f"  {s['vertices']:,} vertices, {s['faces']:,} faces in {t['total']:.1f}s "
              f"(encode {t['encode']:.1f}s, density {t['density']:.1f}s, {s['query_fraction'] * 100:.0f}% of grid evaluated)")
        for w in s["warnings"]:
            print(f"  note: {w}")
        for f, path in files.items():
            print(f"  -> {path}")
    return 1 if fails == len(a.images) else 0


def _studio_progress():
    def cb(stage: str, frac: float, msg: str):
        print(f"  [{max(frac, 0) * 100:3.0f}%] {stage}: {msg}", flush=True)

    return cb


def cmd_studio(a) -> int:
    from .studio import AuraStudio, StudioOptions

    opts = dict(geometry=a.geometry, multiview=a.multiview, quality=a.quality, unwrap=a.unwrap, seed=a.seed,
                reference_weight=a.reference_weight, roughness=a.roughness, metallic=a.metallic,
                min_component=a.min_part_ratio, texture_size=a.texture_size, target_faces=a.faces,
                octree=a.octree, geo_steps=a.steps, mv_steps=a.mv_steps, matting=a.matting, back=a.back,
                best_of=tuple(a.best_of or ()), repair=not a.no_repair, snap=not a.no_snap,
                complete=not a.no_complete, complete_tubes=not a.no_tubes)
    if a.no_normal_map:
        opts["normal_map"] = False
    studio = AuraStudio(options=StudioOptions(**opts), device=a.device, raster=a.raster, weights=a.weights)
    out = Path(a.out)
    fails = 0
    for img in a.images:
        p = Path(img)
        if not p.is_file():
            print(f"skip: {img} (not found)")
            fails += 1
            continue
        print(f"\n{p.name}")
        try:
            res = studio.generate(p, progress=_studio_progress())
        except Exception as e:                                   # keep going with the remaining pictures
            print(f"  failed: {type(e).__name__}: {e}")
            fails += 1
            continue
        files = res.save_all(out / p.stem, "asset", tuple(a.format), up=a.up)
        r = res.report
        print(f"  geometry {r['geometry']} | side views {'yes' if r['side_views'] else 'no'} | "
              f"{r['faces_final']:,} faces | texture {r['texture_size']}px | artwork match IoU {r['reference_iou']} | "
              f"{r['timings']['total']}s")
        for w in r["notes"]:
            print(f"  note: {w}")
        for k, path in files.items():
            print(f"  -> {path}")
    return 1 if fails == len(a.images) else 0


def cmd_setup(a) -> int:
    from .backends.setup_repos import setup_repos

    return setup_repos(a.dir, a.repos, a.zip)


def cmd_serve(a) -> int:
    from .pipeline import AuraWhite
    from .server import serve

    def factory(progress):
        return AuraWhite.from_pretrained(a.weights, a.device, a.precision, trust_pickle=a.trust_checkpoint, progress=progress)

    def studio_factory():
        from .studio import AuraStudio, StudioOptions

        return AuraStudio(options=StudioOptions(), device=a.device, weights=a.weights)

    serve(a.host, a.port, a.out, factory, preload=a.preload, open_browser=not a.no_browser,
          studio_factory=studio_factory)
    return 0


def cmd_fetch(a) -> int:
    from .weights import FP16_NAME, FP32_NAME, convert, download, find_existing_triposr, weights_dir, OFFICIAL_URL

    wd = weights_dir()
    name = FP32_NAME if a.fp32 else FP16_NAME
    final = wd / name
    if final.exists() and not a.force:
        print(f"already prepared: {final}")
        return 0
    src = Path(a.source) if a.source else find_existing_triposr()
    if src:
        print(f"using existing checkpoint: {src}")
    else:
        url = a.url or os.environ.get("AURA_WHITE_WEIGHTS_URL", OFFICIAL_URL)
        print(f"downloading {url}")
        src = download(url, wd / "model.ckpt", _progress_printer())
    print("converting to compact safetensors ...")
    out = convert(src, final, "fp32" if a.fp32 else "fp16", trust_pickle=a.trust_checkpoint)
    print(f"ready: {out} ({out.stat().st_size / 1e9:.2f} GB)")
    if not a.keep_original and src.parent == wd and src.name == "model.ckpt":
        try:
            src.unlink()
            print("removed the downloaded original to save disk space (use --keep-original to keep it)")
        except OSError:
            pass
    return 0


def cmd_convert(a) -> int:
    from .weights import convert

    out = convert(Path(a.src), Path(a.dst) if a.dst else None, "fp32" if a.fp32 else "fp16", trust_pickle=a.trust_checkpoint)
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return 0


def _ver(name: str) -> str:
    try:
        m = __import__(name)
        return getattr(m, "__version__", "installed")
    except Exception as e:
        return f"not available ({type(e).__name__})"


def cmd_doctor(a) -> int:
    print(f"Aura White {__version__}")
    print(f"python   {platform.python_version()}  ({platform.system()} {platform.machine()})")
    print(f"numpy    {_ver('numpy')}")
    print(f"Pillow   {_ver('PIL')}")
    ok = True
    try:
        import torch

        print(f"torch    {torch.__version__}")
        if torch.cuda.is_available():
            print(f"cuda     yes - {torch.cuda.get_device_name(0)}")
        else:
            mps = getattr(torch.backends, "mps", None)
            print("cuda     no (CPU mode)" + ("; Apple MPS available" if mps is not None and mps.is_available() else ""))
    except Exception as e:
        print(f"torch    NOT AVAILABLE: {e}\n         install it: https://pytorch.org/get-started/locally/")
        return 1
    print(f"optional safetensors {_ver('safetensors')} | rembg {_ver('rembg')} | scipy {_ver('scipy')}")
    from .weights import FP16_NAME, FP32_NAME, find_existing_triposr, weights_dir

    wd = weights_dir()
    have = [n for n in (FP16_NAME, FP32_NAME) if (wd / n).exists()]
    print(f"weights  {'ready: ' + str(wd / have[0]) if have else 'not prepared yet'}")
    ex = find_existing_triposr()
    if not have:
        print(f"         {'existing TripoSR checkpoint found: ' + str(ex) + ' (will be re-used, no download)' if ex else 'run `aura-white fetch` once (about 1.7 GB)'}")

    print("\ntextured pipeline (aura-white studio):")
    print(f"  xatlas {_ver('xatlas')} | fast_simplification {_ver('fast_simplification')} | scipy {_ver('scipy')} | "
          f"nvdiffrast {_ver('nvdiffrast')}")
    try:
        from .backends import status_report
        from .raster import get_raster
        from .texture.unwrap import _xatlas_isolated  # noqa: F401

        r = get_raster("auto")
        print(f"  rasteriser: {r.backend} on {r.device}")
        for name, (good, why) in status_report().items():
            print(f"  {'ok ' if good else 'no '} {name:<12} {why}")
        from .texture.unwrap import probe_backend

        used = probe_backend()
        print(f"  UV unwrapper in use: {used}" + ("" if used == "xatlas" else "  (built-in fallback; xatlas missing or unusable)"))
    except Exception as e:
        print(f"  studio check failed: {type(e).__name__}: {e}")

    print("self-test (untrained tiny network, exercises the whole pipeline) ...")
    t0 = time.time()
    try:
        from PIL import Image, ImageDraw

        from .pipeline import AuraWhite

        eng = AuraWhite.from_random()
        im = Image.new("RGB", (160, 160), (235, 235, 235))
        ImageDraw.Draw(im).ellipse((40, 30, 120, 130), fill=(200, 60, 50))
        r = eng.generate(im, resolution=64, threshold="auto", smooth=1)
        with tempfile.TemporaryDirectory() as d:
            files = r.save_all(d, "t", ("glb", "obj", "ply", "stl"))
            assert all(p.stat().st_size > 500 for p in files.values())
            assert files["glb"].read_bytes()[:4] == b"glTF"
        print(f"  PASS  {r.stats['vertices']} vertices, {r.stats['faces']} faces, 4 export formats ({time.time() - t0:.1f}s)")
    except Exception as e:
        ok = False
        print(f"  FAIL  {type(e).__name__}: {e}")
    return 0 if ok else 1


def cmd_forge(a) -> int:
    """GPU-free silhouette-exact relief reconstruction of one picture -> GLB (no textures baked; see `studio`)."""
    from PIL import Image
    from .backends.glb import write_glb_mesh
    from .forge.relief import ReliefGeometry
    from .image.preprocess import prepare_image

    prep = prepare_image(a.image, matting=a.matting, max_side=a.max_side)
    g = ReliefGeometry().generate(prep.cutout, progress=lambda m: print(" ", m))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    v = g.verts[:, [1, 2, 0]]                                  # canonical (z up, viewer +x) -> glTF (y up, front +z)
    write_glb_mesh(out, v, g.faces)
    prep.cutout.save(out.with_suffix(".cutout.png"))
    print(f"wrote {out}  ({len(g.faces):,} faces; chains {g.info['chains']}, cords {g.info['tubes']}); "
          f"matting: {prep.matting}")
    return 0


def cmd_repair(a) -> int:
    from .backends.glb import read_glb, write_glb_mesh
    from .evaluation import load_mesh
    from .mesh.repair import RepairOptions, repair_mesh

    v, f = load_mesh(a.mesh)
    v2, f2, rep = repair_mesh(v, f, RepairOptions(max_hole_edges=a.max_hole_edges, min_island_faces=a.min_island_faces))
    out = a.out or str(Path(a.mesh).with_suffix("")) + "_repaired.glb"
    write_glb_mesh(out, v2, f2)
    import json as _json
    print(_json.dumps({k: rep[k] for k in rep}, indent=2, default=str))
    print(f"wrote {out}")
    return 0


def cmd_eval(a) -> int:
    from .evaluation import evaluate_dirs, save_report

    rep = evaluate_dirs(a.pred, a.gt, tau=a.tau)
    out = a.out or str(Path(a.pred) / "eval_report.json")
    save_report(rep, out)
    s = rep["summary"]
    print(f"{rep['n_pairs']} pairs ({rep['n_failed']} failed): " + ", ".join(f"{k}={v:.4f}" for k, v in s.items()))
    print(f"report -> {out}")
    return 0 if rep["n_pairs"] else 1


def cmd_shape_train(a) -> int:
    from .shape.train import main as train_main

    argv = ["--steps", str(a.steps), "--res", str(a.res), "--patch", str(a.patch), "--d-model", str(a.d_model),
            "--layers", str(a.layers), "--heads", str(a.heads), "--batch", str(a.batch), "--lr", str(a.lr),
            "--data", a.data, "--out", a.out, "--device", a.device]
    if a.resume:
        argv += ["--resume", a.resume]
    return train_main(argv)


def cmd_shape_complete(a) -> int:
    from .backends.glb import write_glb_mesh
    from .evaluation import load_mesh
    from .shape.complete import ShapeCompleter

    v, f = load_mesh(a.mesh)
    comp = ShapeCompleter.load(a.ckpt, a.device)
    v2, f2 = comp.complete_mesh(v, f, steps=a.steps, seed=a.seed)
    out = a.out or str(Path(a.mesh).with_suffix("")) + "_completed.glb"
    write_glb_mesh(out, v2[:, [1, 2, 0]], f2)
    print(f"wrote {out}  ({len(f2):,} faces, {comp.res}^3 volume)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aura-white", description="Aura White - lightweight single-image 3D generation")
    p.add_argument("--version", action="version", version=f"aura-white {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--weights", help="path to weights (.safetensors / model.ckpt); default: automatic")
        sp.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1 | mps")
        sp.add_argument("--precision", default="auto", help="auto (= fp32) | fp32 | fp16 (low VRAM, CUDA) | bf16 (fast on modern CPUs/GPUs)")
        sp.add_argument("--trust-checkpoint", action="store_true", help="allow unsafe pickle loading for a checkpoint you trust")

    st = sub.add_parser("studio", help="image(s) -> textured 3D asset (UV albedo + normal map) - the sharp pipeline")
    st.add_argument("images", nargs="+")
    st.add_argument("-o", "--out", default="aura_out")
    st.add_argument("--geometry", default="auto",
                    choices=["auto", "pixal3d", "trellis2", "hunyuan3d", "instantmesh", "relief", "aura"],
                    help="shape model (auto: Pixal3D > TRELLIS.2 > Hunyuan3D-2.1 > InstantMesh > built-in, whichever can "
                         "run here; relief = GPU-free silhouette-exact reconstruction for thin/flat subjects)")
    st.add_argument("--best-of", nargs="+", default=None, metavar="BACKEND",
                    help="run several shape models and keep the one whose silhouette matches the artwork best "
                         "(e.g. --best-of pixal3d trellis2 hunyuan3d)")
    st.add_argument("--matting", default="auto", choices=["auto", "exact", "rembg", "border"],
                    help="background removal (auto: exact geometric matte for flat backgrounds, else rembg)")
    st.add_argument("--back", default="auto", choices=["auto", "mirror", "off"],
                    help="texture the hidden back with the mirrored artwork (auto: relief geometry only)")
    st.add_argument("--no-repair", action="store_true", help="skip topology repair")
    st.add_argument("--no-snap", action="store_true", help="skip pulling the mesh outline onto the artwork outline")
    st.add_argument("--no-complete", action="store_true", help="skip building parts the mesh is missing (chains, ribbons, ...)")
    st.add_argument("--no-tubes", action="store_true", help="completion: chains only, no round cords")
    st.add_argument("--multiview", default="auto", choices=["auto", "on", "off"], help="Zero123++ side views for hidden surfaces")
    st.add_argument("--quality", default="high", choices=["draft", "standard", "high", "ultra"])
    st.add_argument("--texture-size", type=int, help="atlas resolution (default from quality: 1024/2048/4096)")
    st.add_argument("--faces", type=int, help="final triangle count (default from quality)")
    st.add_argument("--octree", type=int, help="Hunyuan3D shape resolution (256-512)")
    st.add_argument("--steps", type=int, help="shape diffusion steps")
    st.add_argument("--mv-steps", type=int, help="Zero123++ diffusion steps")
    st.add_argument("--no-normal-map", action="store_true")
    st.add_argument("--unwrap", default="auto", choices=["auto", "xatlas", "builtin"])
    st.add_argument("--raster", default="auto", choices=["auto", "torch", "nvdiffrast"])
    st.add_argument("--reference-weight", type=float, default=8.0, help="how strongly the original artwork wins over generated views")
    st.add_argument("--roughness", type=float, default=0.7)
    st.add_argument("--metallic", type=float, default=0.0)
    st.add_argument("--min-part-ratio", type=float, default=0.002, help="drop floating pieces smaller than this fraction of the main body")
    st.add_argument("--format", nargs="+", default=["glb", "obj"], choices=["glb", "obj"])
    st.add_argument("--up", default="y", choices=["y", "z"])
    st.add_argument("--seed", type=int, default=42)
    st.add_argument("--device", default="auto")
    st.add_argument("--weights", default=None, help="Aura White weights for --geometry aura (default: auto)")
    st.set_defaults(fn=cmd_studio)

    su = sub.add_parser("setup", help="download the upstream model repositories (InstantMesh, Hunyuan3D-2.1, Zero123++) into third_party/")
    su.add_argument("--dir", default=None, help="target folder (default: <project>/third_party)")
    su.add_argument("--repos", nargs="+", default=["zero123plus", "instantmesh", "hunyuan3d"],
                    choices=["zero123plus", "instantmesh", "hunyuan3d", "trellis2", "pixal3d", "nvdiffrast"])
    su.add_argument("--zip", nargs="*", default=[], help="use zip files you already have instead of downloading")
    su.set_defaults(fn=cmd_setup)

    g = sub.add_parser("generate", help="image(s) -> 3D mesh files (vertex colours, fast)")
    g.add_argument("images", nargs="+")
    g.add_argument("-o", "--out", default="aura_out")
    g.add_argument("--format", nargs="+", default=["glb", "obj"], choices=["glb", "obj", "ply", "stl"])
    g.add_argument("--resolution", type=int, default=256, help="grid resolution (64-512); higher = more detail, slower")
    g.add_argument("--threshold", default="25", help="surface density threshold (default 25 = TripoSR)")
    g.add_argument("--smooth", type=int, default=2, help="Taubin smoothing iterations (0 = off)")
    g.add_argument("--min-part-ratio", type=float, default=0.02, help="drop floating pieces smaller than this fraction of the main body (0 = keep all)")
    g.add_argument("--no-remove-bg", action="store_true", help="input is already cut out / has a clean background")
    g.add_argument("--matting", default="auto", choices=["auto", "rembg", "border"])
    g.add_argument("--foreground-ratio", type=float, default=0.85)
    g.add_argument("--dense", action="store_true", help="evaluate the full grid instead of the adaptive shell (slower, reference)")
    g.add_argument("--up", default="y", choices=["y", "z"], help="output up axis (y = glTF/Blender-import friendly)")
    g.add_argument("--yaw", type=float, default=0.0, help="extra rotation about the up axis, degrees")
    g.add_argument("--turntable", action="store_true", help="also render a turntable GIF (slow on CPU)")
    common(g)
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("serve", help="local web app with 3D viewer")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("-o", "--out", default="aura_out")
    s.add_argument("--preload", action="store_true", help="load the model before accepting requests")
    s.add_argument("--no-browser", action="store_true")
    common(s)
    s.set_defaults(fn=cmd_serve)

    f = sub.add_parser("fetch", help="prepare weights (re-uses an existing TripoSR download, else downloads once)")
    f.add_argument("--source", help="a model.ckpt you already have")
    f.add_argument("--url")
    f.add_argument("--fp32", action="store_true", help="keep full precision (2x larger; fp16 is visually identical)")
    f.add_argument("--force", action="store_true")
    f.add_argument("--keep-original", action="store_true")
    f.add_argument("--trust-checkpoint", action="store_true")
    f.set_defaults(fn=cmd_fetch)

    c = sub.add_parser("convert", help="convert a checkpoint to compact .safetensors")
    c.add_argument("src")
    c.add_argument("dst", nargs="?")
    c.add_argument("--fp32", action="store_true")
    c.add_argument("--trust-checkpoint", action="store_true")
    c.set_defaults(fn=cmd_convert)

    fg = sub.add_parser("forge", help="picture -> silhouette-exact relief mesh with real chains (no GPU, no model download)")
    fg.add_argument("image")
    fg.add_argument("-o", "--out", default="aura_out/relief.glb")
    fg.add_argument("--matting", default="auto", choices=["auto", "exact", "rembg", "border"])
    fg.add_argument("--max-side", type=int, default=2048)
    fg.set_defaults(fn=cmd_forge)

    rp = sub.add_parser("repair", help="fix non-manifold edges/vertices, flipped faces, holes and dust in a .glb/.obj")
    rp.add_argument("mesh")
    rp.add_argument("-o", "--out", default=None)
    rp.add_argument("--max-hole-edges", type=int, default=400)
    rp.add_argument("--min-island-faces", type=int, default=24)
    rp.set_defaults(fn=cmd_repair)

    ev = sub.add_parser("eval", help="chamfer / F-score / normal consistency / voxel IoU for folders of matching meshes")
    ev.add_argument("--pred", required=True)
    ev.add_argument("--gt", required=True)
    ev.add_argument("--tau", type=float, default=0.02)
    ev.add_argument("--out", default=None)
    ev.set_defaults(fn=cmd_eval)

    stn = sub.add_parser("shape-train", help="train Aura Shape (flow-matching DiT shape completion) from scratch")
    stn.add_argument("--steps", type=int, default=20000)
    stn.add_argument("--res", type=int, default=32)
    stn.add_argument("--patch", type=int, default=4)
    stn.add_argument("--d-model", type=int, default=384)
    stn.add_argument("--layers", type=int, default=12)
    stn.add_argument("--heads", type=int, default=6)
    stn.add_argument("--batch", type=int, default=16)
    stn.add_argument("--lr", type=float, default=2e-4)
    stn.add_argument("--data", default="procedural", help="'procedural' or a folder of .glb/.obj meshes")
    stn.add_argument("--out", default="aura_shape.pt")
    stn.add_argument("--resume", default=None)
    stn.add_argument("--device", default="auto")
    stn.set_defaults(fn=cmd_shape_train)

    scm = sub.add_parser("shape-complete", help="complete the hidden side of a single-view mesh with an Aura Shape checkpoint")
    scm.add_argument("mesh")
    scm.add_argument("--ckpt", required=True)
    scm.add_argument("-o", "--out", default=None)
    scm.add_argument("--steps", type=int, default=32)
    scm.add_argument("--seed", type=int, default=0)
    scm.add_argument("--device", default="auto")
    scm.set_defaults(fn=cmd_shape_complete)

    d = sub.add_parser("doctor", help="check this Python/PyTorch install and run a self-test")
    d.set_defaults(fn=cmd_doctor)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "generate")  # `aura-white photo.png` just works
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        print("\ncancelled")
        return 130
    except Exception as e:  # readable one-line errors instead of tracebacks
        if os.environ.get("AURA_WHITE_DEBUG"):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
