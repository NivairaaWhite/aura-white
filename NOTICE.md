# Notices

Aura White re-implements, in plain PyTorch, the network architecture published as **TripoSR**
(Tochilkin et al., Tripo AI and Stability AI, 2024; https://github.com/VAST-AI-Research/TripoSR,
MIT licence). The pretrained weights it loads are the official `stabilityai/TripoSR` checkpoint
(MIT licence) and are **not** included in this package - `aura-white fetch` obtains them (or re-uses
a copy you already have).

The transformer block layout follows the Hugging Face `diffusers` design (Apache-2.0) and the image
encoder is the DINO ViT-B/16 architecture (Apache-2.0, Meta AI); both are written from scratch here so
that no `diffusers` / `transformers` install is needed.

Keep these credits if you redistribute the weights or derived work.

## Textured pipeline (`aura-white studio`)

The studio orchestrates - it does **not** bundle - these third-party projects. You unpack or download them yourself
(`aura-white setup`), and their licences apply to what you generate with them:

* **Hunyuan3D-2.1** (Tencent) - Tencent Hunyuan 3D 2.1 Community License (territory and usage conditions apply; read it
  before commercial use). https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
* **InstantMesh** (Tencent ARC) - code Apache-2.0; weights Apache-2.0. https://github.com/TencentARC/InstantMesh
* **Zero123++** (SUDO-AI) - code Apache-2.0; **weights CC-BY-NC 4.0 (non-commercial)**. https://github.com/SUDO-AI-3D/zero123plus
  Textures from Zero123++ side views inherit that restriction. Run with `--multiview off` to avoid it.
* **nvdiffrast** (NVIDIA) - NVIDIA Source Code License (non-commercial research use). Optional; Aura White's own
  PyTorch rasteriser is the default.
* **xatlas** (via xatlas-python, MIT) and **fast-simplification** (MIT): optional, each with a built-in fallback.
