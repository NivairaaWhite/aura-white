# Aura White

Single-image 3D generation, rebuilt from TripoSR to **keep working when Python moves on**.
Photo in -> coloured 3D mesh out (GLB, OBJ, PLY, STL), from the command line, a Python API, or a local
web app with a built-in 3D viewer.

Only three packages are required: **numpy, Pillow, torch**. No compiler, no git dependencies, no pinned
versions, no `transformers`, no `gradio`.

## Aura White Studio (v3): exact-to-the-pixel, with parts other systems drop

The original pipeline colours *vertices*, so fine detail is smeared into "melted clay". **`aura-white studio`** builds
what commercial services do - a UV-unwrapped mesh with a 2K-4K albedo texture and a normal map - from open models,
puts the **original artwork's own pixels** on every surface it can see, then **repairs, snaps and completes** the
result so hairline chains, cords, ribbons and gems that every neural model drops are no longer missing:

```
artwork ─┬─> matte       exact geometric matte (flat backgrounds incl. white-on-white) | rembg | transparent PNG as-is
         ├─> geometry    Pixal3D | TRELLIS.2 | Hunyuan3D-2.1 | InstantMesh | relief (GPU-free) | built-in Aura White
         │               (auto: the best that can run here; --best-of runs several and keeps the one whose
         │                silhouette matches the artwork best)
         ├─> side views  Zero123++ (only for the far side of the object)
         └─> registration: the artwork's camera is recovered by silhouette matching + stereo consistency
   mesh ── topology repair ── silhouette snap ── residual completion (chains/cords/ribbons/gems) ──
        decimate ── UV unwrap ── multi-view bake (+ mirrored-artwork back) ── normal map ── GLB / OBJ+MTL
```

```
run.bat studio art.png                              # auto geometry, auto matting, repair+snap+complete all on
run.bat studio art.png --best-of pixal3d trellis2 hunyuan3d --quality high
run.bat studio art.png --geometry relief             # GPU-free, silhouette-exact - best for flat/thin subjects
                                                      # (weapons, wings, emblems, cut-out characters); real
                                                      # interlocking chain links and round cords, not blobs
run.bat studio art.png --matting exact               # force the geometric matte (keeps hairlines, no ML model)
run.bat studio art.png --no-complete --no-snap        # geometry + texture only, no post-processing
run.bat setup --repos pixal3d trellis2 hunyuan3d      # clone the upstream repos into third_party/
```

**Why chains stopped disappearing.** Every feed-forward image-to-3D network (TripoSR, TRELLIS, Hunyuan3D, ...) drops a
2-pixel-wide chain: a marching-cubes/FlexiCubes threshold cannot hold a feature that thin, and the mesh-cleanup pass
that keeps output watertight throws small disconnected parts away as noise. Instead of trying to force a diffusion
model to hallucinate correct topology from one picture, Aura White finds what the mesh is missing directly in the
**artwork's own camera** (`aura_white/forge/complete.py`): pixels the artwork shows that no mesh surface projects onto.
A skeleton-based detector (`forge/wires.py`) tells a hairline chain from a ribbon or a fin by how its inscribed radius
varies along its length, and builds **real interlocking stadium links** (or a round tapered cord for laces/vines) at
the correct depth and width; everything else missing (gems, studs, small fins) becomes a silhouette-exact relief patch
(`forge/patch.py`, marching-squares + Delaunay, holes included). `--geometry relief` uses the same engine as the whole
shape, for subjects (weapons, banners, emblems, 2D characters used as props) where an exact 2.5-D reconstruction beats
an approximate 3-D guess - no GPU, no model download, runs in a couple of seconds.

**Registration is then used twice more.** `forge/snap.py` pulls the mesh's own silhouette onto the artwork's silhouette
(within the image plane only, relaxed smoothly over the surface) before texturing, so a blade tip or a fingertip a
neural model rounded off lines up exactly with the picture. `mesh/repair.py` is a from-scratch, fully vectorised
topology fixer (no trimesh/pymeshlab/open3d): non-manifold edges are cut by keeping the most-coplanar pair of faces
and splitting the rest, "bow-tie" vertices are split by connected-components over face corners, boundary loops are
closed with ear-clipping, orientation is made consistent per shell by BFS + signed volume, and dust islands are
dropped - all reported in `*_report.json["refine"]` so a bad repair is visible, never silent.

Why it stays crisp: the bake weights the original picture 8x over generated views, so wherever the artwork can see the
surface the texture *is* the artwork (full resolution, exact colours); Zero123++ views (only 320 px) fill in what the
artwork cannot see, colour-matched to it; for the `relief` backend the far side is instead textured with the artwork
**mirrored** across the object's own depth axis (`--back mirror`, on by default for `relief`) rather than blurred
nearest-neighbour fill, which is the correct answer for the many game-art weapons/props that are left-right symmetric
through their thickness. Texels nothing can see are completed from their nearest coloured neighbours in 3D (seamless
across UV islands). Silhouette/occlusion-aware weights stop background colour bleeding at the edges.

**Never breaks:** the texturing engine (`aura_white/raster`, `aura_white/texture`) is pure PyTorch/NumPy, and the new
`forge`/`mesh.repair`/`evaluation` modules are pure NumPy/SciPy - no compiled extension, no native wheel, nothing that
can segfault the interpreter. `xatlas` and `fast-simplification` still run **in a child process** and fall back to the
built-in equivalents on any misbehaviour. Pixal3D and TRELLIS.2 run **as a subprocess in their own Python
environment** (`AURA_WHITE_PIXAL3D_PYTHON` / `AURA_WHITE_TRELLIS2_PYTHON`), so their pinned CUDA/torch versions can
never conflict with Aura White's or with each other's; a crash there is caught, reported, and the pipeline falls back
to the next geometry backend. Every heavy model is optional, lazily imported, loaded, used and freed in turn.

Colab (free T4 works with `--quality standard`): open `notebooks/aura_white_studio_colab.ipynb`.
Extras: `pip install -r requirements-studio.txt` (or `install.bat cu126 studio`). Sources of the model repositories go in
`third_party/` (see `third_party/README.md`); Pixal3D and TRELLIS.2 keep their own virtualenv (`setup.sh` in each repo)
since they pin exact CUDA/torch versions - point `AURA_WHITE_PIXAL3D_DIR`/`_PYTHON` (and `_TRELLIS2_`) at them once
built. **Licences:** Zero123++ weights are non-commercial (CC-BY-NC) - use `--multiview off` for commercial work;
Hunyuan3D-2 is usable commercially outside the EU/UK/South Korea; TRELLIS.2 and TripoSR are MIT. See `NOTICE.md`.

Python:

```python
from aura_white.studio import AuraStudio, StudioOptions
studio = AuraStudio(options=StudioOptions(quality="high", best_of=("pixal3d", "trellis2", "hunyuan3d")))
result = studio.generate("art.png", progress=lambda stage, frac, msg: print(stage, msg))
result.save_all("out", "hero", formats=("glb", "obj"))         # + albedo/normal PNGs, previews, report.json
print(result.report["reference_iou"], result.report["notes"])  # how well the artwork matched the mesh, fallbacks taken
print(result.report["refine"])                                 # repair / snap / completion stats
```

The `*_report.json` and `*_registration.png` (artwork vs mesh silhouette: red = mesh sticks out, blue = artwork not covered)
tell you at a glance whether a result can be trusted.

### Other new commands

```
run.bat forge art.png -o out.glb           # relief backend alone: picture -> exact mesh in ~1-3s, no GPU
run.bat repair mesh.glb                    # fix non-manifold edges/vertices, holes, dust in any mesh
run.bat eval --pred outputs --gt ground_truth   # chamfer / F-score / normal consistency / voxel IoU, batch report
run.bat shape-train --steps 20000 --data procedural   # train Aura Shape (see below) from scratch, no dataset needed
run.bat shape-train --data /path/to/glb_or_obj_folder  # ...or on your own mesh collection (Objaverse, etc.)
run.bat shape-complete mesh.glb --ckpt aura_shape.pt   # fill in the hidden volume of a single-view mesh
```

### Aura Shape: an owned model for volume no picture shows

`aura_white/shape/` is a small conditional **flow-matching Diffusion Transformer over a truncated-SDF volume**
(`ShapeDiT` + rectified-flow training/sampling, `model.py`), trained from scratch - no borrowed weights. Given the part
of an object's volume a single view can observe, it generates the rest, with the observed voxels held exactly
consistent throughout sampling (`FlowMatching.sample`). It trains on an unlimited built-in procedural-shape generator
(random unions of primitives, `shape/sdf.py`) so `shape-train --data procedural` needs no dataset at all, or on any
folder of `.glb`/`.obj` meshes (Objaverse, your own asset library, ...) via `--data <folder>`. This is a genuinely
different tool from the residual completion above: completion rebuilds parts *visible in the picture* that the mesh
lost; Aura Shape guesses volume *no picture shows at all*. Both are optional and off by default in the main pipeline;
use `shape-complete` directly, or call `aura_white.shape.ShapeCompleter` from Python.

## What broke in TripoSR, and what replaced it

| TripoSR (original)                                   | Problem on a current Python stack                                        | Aura White |
|------------------------------------------------------|--------------------------------------------------------------------------|------------|
| `torchmcubes` (git, compiled)                        | Abandoned; needs a compiler; fails to build on new Python                | Built-in **surface-nets mesher** (numpy, ~100 lines) |
| `transformers==4.35.0`                               | Newer releases renamed the ViT layers, so the official checkpoint no longer loads | Own DINO ViT in plain PyTorch + a key normaliser (accepts old *and* new names) |
| `Pillow==10.1.0`, `omegaconf`, `einops`, `trimesh==4.0.5`, `xatlas==0.0.9`, `moderngl==5.10.0` | Exact pins / compiled wheels that don't exist for new Pythons | Not needed (own exporters, no UV baking) |
| `gradio`                                             | API changes every release                                                | Standard-library web server + WebGL viewer |
| `rembg` required                                     | onnxruntime/numba lag behind new Pythons                                 | Optional; built-in matte for plain backgrounds; transparent PNGs used as-is |
| `torch.cross(a, b)` without `dim`                    | Picks the wrong axis when there happen to be exactly 3 views             | Explicit `dim=-1` (`torch.linalg.cross`) |

Every version-sensitive spot lives in `aura_white/_compat.py` behind feature detection.

## Better than the original

* **No compiled parts.** Nothing to build, nothing that can fail to build.
* **Watertight, manifold, outward-facing meshes** (surface nets + checkerboard-face repair), with fewer, better-shaped triangles than marching cubes. Mesh is closed even if the object touches the bounding box.
* **5-10x fewer network queries**: a coarse pass finds the surface shell and only that shell is evaluated at full resolution (`--dense` gives the reference behaviour).
* **Floating specks removed**, light volume-preserving smoothing, degenerate faces dropped.
* **Half-size, pickle-free weights**: one-time conversion to fp16 `.safetensors` (about 0.84 GB instead of 1.68 GB). Re-uses a TripoSR checkpoint already on your PC (Hugging Face cache) - no second download.
* **Deterministic**, thread-safe API, `--low-memory` mode, CPU / CUDA / Apple MPS, fp32 (default) / fp16 / bf16.
* **`aura-white doctor`**: checks your install and runs a full self-test in seconds, so a Python/PyTorch update that breaks something is caught immediately.

## Install

**Windows:** double-click `install.bat` (or run it in a terminal).
NVIDIA GPU: `install.bat cu126` (use the tag shown on https://pytorch.org/get-started/locally/ for your CUDA version).
**Linux / macOS:** `./install.sh` (or `./install.sh cu126`).

Manual: `pip install -e ".[recommended]"` inside any virtual environment with Python 3.9 or newer.

Then check it: `run.bat doctor` (Windows) / `.venv/bin/aura-white doctor`.

## Use

```
run.bat photo.png                       # -> aura_out/photo/mesh.glb + mesh.obj
run.bat photo.png other.jpg --format glb obj ply stl --resolution 320
run.bat serve                           # local web app at http://127.0.0.1:7860 with a 3D viewer
```

The first real run prepares the weights automatically (or run `run.bat fetch`). If you ever used
TripoSR before, its `model.ckpt` is found and reused; otherwise about 1.7 GB is downloaded once.
Use `--weights PATH` to point at any `model.ckpt` / `.safetensors` yourself.

Python:

```python
from aura_white import AuraWhite

engine = AuraWhite.from_pretrained()                     # device="auto", precision="auto" (= fp32)
result = engine.generate("photo.png", resolution=256)    # path, bytes, PIL image or array
result.save("out.glb")                                   # or .obj / .ply / .stl
print(result.stats["vertices"], result.timings["total"])
```

### Options (`generate`)

| option | default | meaning |
|---|---|---|
| `--resolution` | 256 | grid size (64-512). Higher = more detail, slower |
| `--threshold` | 25 | density level of the surface (TripoSR default). Lower = fatter, higher = thinner; "No surface found" -> lower it |
| `--smooth` | 2 | Taubin smoothing passes (0 = off) |
| `--min-part-ratio` | 0.02 | drop floating pieces smaller than this share of the main body (0 keeps everything) |
| `--no-remove-bg` | off | image is already cut out / clean |
| `--matting` | auto | `auto` (alpha -> rembg -> built-in) / `rembg` / `border` |
| `--foreground-ratio` | 0.85 | how much of the frame the object fills |
| `--dense` | off | evaluate the whole grid (reference, slower) |
| `--up`, `--yaw` | y, 0 | output axis convention and extra rotation about it |
| `--device`, `--precision` | auto, auto | `cpu`/`cuda`/`mps`; `fp32`/`fp16`/`bf16` (fp16 = lower VRAM, CUDA) |
| `--turntable` | off | also render a turntable GIF (slow on CPU) |

Photos work best with one object, plain background, three-quarter view. Transparent PNGs skip background removal entirely.

## What has been verified (and what has not)

Verified by the test suite (`python -m pytest`, 71 tests) and `tools/parity_check.py`:

* **Architecture equals TripoSR's**: at full size both networks have 547 tensors / 418.7 M parameters with identical names and shapes (built from the published `config.yaml` values), so the official checkpoint loads without renaming.
* **Numerically equal to the original code**: with identical random weights, scene codes, densities, colours and rendered views match the original TripoSR source to float precision (tiny-size model; includes the position-embedding resize path). Perturbing any stage is detected, so the comparison is not vacuous.
* **Mesher**: sphere/torus/noise-field tests for closed, consistently oriented, edge-manifold output; adaptive evaluation reproduces dense evaluation (identical inside/outside classification on analytic scenes) while using about 9% of the queries at 256.
* **Pipeline end to end** with a random tiny network: preprocessing, encoding, meshing, colouring, all four export formats (read back with `trimesh`), CLI, web server job flow, weights conversion / loading (pickle, safetensors, fp16), resumable download.
* **Python versions**: the numpy-side (mesher, volume, image, exporters; 47 tests) passes on Python 3.9, 3.12, 3.13 and 3.14; the PyTorch side was run on Python 3.12 with PyTorch 2.14 / NumPy 2.5 / Pillow 12.

Not verified in the build environment (no access to the weights or a GPU there):

* Output quality with the **real pretrained weights** - run `python tools/parity_check.py --triposr <TripoSR-main folder> --real <folder with config.yaml + model.ckpt> --image photo.png` to compare against the original code, or just compare a mesh from both.
* The default `--up y` orientation puts the side the photo was taken from toward +Z; if a model comes out facing away, use `--yaw 180`.
* The web viewer's WebGL rendering in a real browser (its data format and script were checked, but not drawn).
* PyTorch on Python 3.13/3.14, CUDA, and Apple MPS.

## Layout

```
aura_white/
  model/      dino.py transformer.py triplane.py renderer.py net.py   (pure PyTorch, TripoSR-compatible)
  mesh/       surface_nets.py cleanup.py export.py repair.py          (numpy/scipy only; repair.py: topology fixer)
  shape/      sdf.py model.py complete.py train.py                    (Aura Shape: owned flow-matching DiT for hidden volume)
  forge/      morph.py patch.py wires.py complete.py snap.py softraster.py relief.py   (image-space exact geometry; numpy/scipy only)
  image/      preprocess.py matte.py                                  (Pillow + numpy; matte.py: exact geometric matting)
  volume.py   coarse-to-fine density evaluation
  pipeline.py AuraWhite / MeshResult          weights.py  find, download, convert
  server.py viewer.py  web app                cli.py      studio | generate | serve | fetch | convert | setup | doctor |
                                                          forge | repair | eval | shape-train | shape-complete
  raster/     torch_raster.py (exact z-buffer, no extensions) nvdiff.py (optional) camera.py
  texture/    unwrap decimate register bake fill normals preview  (UV atlas, camera fitting, multi-view baking)
  backends/   geometry.py (selection/auto/best-of) external.py (Pixal3D, TRELLIS.2: own-env subprocess) glb.py
              (dependency-free glTF reader/writer) hunyuan3d.py instantmesh.py multiview.py (Zero123++) setup_repos.py
  evaluation.py   chamfer / F-score / normal consistency / voxel IoU / silhouette IoU, batch driver
  studio.py   AuraStudio: the whole textured pipeline      _isolate.py  run fragile native code in a child process
tools/parity_check.py   comparison against the original TripoSR source
tests/                  pytest suite
```

## Troubleshooting

* `torch NOT AVAILABLE` in `doctor`: install PyTorch for your Python from https://pytorch.org/get-started/locally/.
* "No surface found": lower `--threshold` (try 10), or use a clearer photo.
* Rough cut-out on a busy background: use a transparent PNG, or `pip install rembg`.
* Out of memory: add `--low-memory` (Python: `low_memory=True`) or lower `--resolution`.
* Set `AURA_WHITE_DEBUG=1` to see full tracebacks; `AURA_WHITE_HOME` moves the weights cache.

## Credits and licence

MIT licence (see `LICENSE`). Network design and weights: TripoSR by Tripo AI and Stability AI (MIT); see `NOTICE.md`.
