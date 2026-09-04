# FoV-Net — LLM Agent Context

> Context file for AI coding agents working in this repository.
> Companion paper: `docs/FoV-Net.pdf` — *"FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting"* (Ballegeer & Benoit, Ghent University, CVPR 2026, arXiv:2602.24084).

## 1. What this project is

FoV-Net is a neural network for learning directly from **CAD Boundary Representations (B-reps)** — STEP files — for whole-model **classification** and per-face **segmentation**. Its core contribution is **rotation invariance**: prior UV-grid-based methods (UV-Net, AAGNet) encode absolute coordinates/normals and collapse from >95% to as low as ~10% accuracy under arbitrary SO(3) rotations. FoV-Net stays identical under rotation by construction (it is also translation-invariant), and is markedly more data-efficient in low-data regimes.

Each B-rep model becomes a **face-adjacency graph** (faces = nodes, shared boundary curves = edges). Every face gets three rotation-invariant descriptors:

1. **LRF UV-grid** (`x_local`, 10×10×7) — points, normals, and trim mask sampled in the face's UV parametric domain, expressed in a per-face **Local Reference Frame** `R_f = [U V N]`: `N` = outward normal at the face center `o` (surface point at the UV-domain center), `U` = U-direction tangent projected onto the tangent plane and normalized, `V = N × U`. Transform: `p' = R_fᵀ(p − o)`, `n' = R_fᵀ n`. Since `o` and `R_f` co-rotate with the face, identical faces in different poses yield identical grids.
2. **Field-of-View (FoV) grids** (`vision_grids`, el×az×6) — rays cast from `o` over a hemisphere around `+N` (**Outward Vision, OV** — external surroundings) and around `−N` (**Inward Vision, IV** — interior of the solid). The hemisphere is discretized into elevation (0–90°) × azimuth (0–360°) bins; azimuth 0° is aligned with `U`, increasing toward `V` (so directions are also expressed in the LRF → invariant). Per ray, three channels at the first hit: **occupancy** (hit flag), **distance**, and **dot** (ray direction · hit-surface normal, i.e. incidence angle). Misses are all-zero. Default resolution 6×12 (15° elevation, 30° azimuth steps).
3. **Face attributes** (`face_feat`, 7D) — surface-type flags + area (exact order in §5).

Per-face CNN embeddings are fused by an MLP and propagated over the B-rep graph by a 3-layer **GAT** (4 heads). Classification: max-pool over nodes → MLP head. Segmentation: per-node head on `[node_emb ‖ graph_emb]`.

**Key paper results** (mean over 5 seeds; rotated / original test):
- SolidLetters classification: FoV-Net **96.35 / 96.35**; UV-Net 8.94 / 97.10; AAGNet 14.03 / 96.68. Rotation-augmented baselines reach ~94–95 but lose original-set accuracy.
- MFCAD++ segmentation IoU: FoV-Net **97.81 / 97.81**; UV-Net 18.79 / 96.65.
- Ablation (SolidLetters, single feature only): FoV-only 95.79, LRF-UV-only 94.39, IV-only 93.40, OV-only 92.92, face-feat-only 70.68, topology-only 37.72. FoV resolution degrades gracefully down to 2×4; a single ray (1×1) collapses to 75.04.
- Data efficiency (MFCAD++): 80% accuracy with 50 training samples vs ~60% (AAGNet) / ~30% (UV-Net).

## 2. Repository map

```
fovnet/
├── fovnet/
│   ├── FOVNet.py             # FOVNet nn.Module + FOVNetModule (Lightning wrapper)
│   └── encoders.py           # SurfaceEncoder, VisionGridEncoder, GraphEncoder (GAT), fusion MLP
├── preprocessing/
│   ├── preprocess.py         # STEP → DGL graph pipeline (CLI, multiprocessing, ProcessingConfig)
│   ├── geometry_features.py  # UV grids, LRF computation, face attributes, seg-label extraction
│   └── ray_casting.py        # Hemisphere sampling; B-rep (OCC) and mesh (trimesh/embree) ray casting
├── datasets/
│   ├── base.py               # BaseDataset: lazy .bin loading, transforms, collation
│   ├── solidletters.py       # 26-class classification
│   ├── traceparts.py         # 6-class classification
│   ├── fusion360.py          # 8-class segmentation
│   ├── mfcad.py              # 25-class segmentation (MFCAD++)
│   ├── bendfm.py             # 2-class sheet-metal manufacturability (NOT in main.py registry)
│   └── util.py               # bbox/rotation/file helpers
├── visuals/visualize_rays.py # Ray visualization
├── main.py                   # Train/test entry point (Lightning), dataset registry, CLI
├── pixi.toml / pixi.lock     # Environment (primary); environment.yaml is conda fallback
└── docs/FoV-Net.pdf          # The paper; docs/fovnet_architecture.jpg = Fig. 6
```

## 3. Environment & commands

Managed with **pixi** (linux-64; Python 3.9, PyTorch 2.4 + CUDA 12.4, DGL 2.4+cu121, Lightning 2.5, pythonocc-core 7.5.1, occwl, trimesh+embreex). Prefix everything with `pixi run` (or `pixi shell` once).

```bash
pixi install                                                     # create env from lockfile
pixi run preprocess --dataset solidletters --folder graphs --all # STEP → .bin graphs
pixi run train --dataset solidletters --graph_path graphs        # train + auto-test best ckpt
pixi run test  --dataset solidletters --ckpt checkpoints/best.ckpt --graph_path graphs
pixi run lint                                                    # ruff check . (config: ruff.toml)
```

Useful flags:
- `preprocess.py`: `--split train|val|test` (or `--all`), `--az/--el` (FoV resolution), `--uv_samples`, `--rotate` (writes `<split>_rotated` graphs + `*_rotated.stp`, deterministic per-file rotation), `--mesh_rays` (fast tessellation-based casting), `--skip_existing`, `--no_compress`, `--edge_info`, `--num_processes`.
- `main.py`: `--aug` (SO(3) rotation augmentation at load time), `--lc N` (learning-curve subsampling, stratified for classification), `--global_uv` (use `x` instead of `x_local` — breaks rotation invariance, this is the "FoV-Net (UV)" paper variant), ablations `--no_vision --no_ov --no_iv --no_uv --no_face_feat`, `--wandb`.

Training loop (`main.py`): AdamW (lr 1e-3), batch 64, early stopping on `val_loss` (patience 25, min_delta 1e-3), best checkpoint by `val_loss` under `checkpoints/<MMDD>/<HHMM>/`, deterministic seeding (seed 42), TensorBoard always, then evaluates both `test` and `test_rotated` splits.

## 4. Data layout & preprocessing pipeline

Expected on-disk layout (raw STEP + generated graphs):

```
data/<dataset>/<split>/*.step            # splits: train, val, test (+ test_rotated generated)
data/<dataset>/<split>/graphs/*.bin      # DGL graphs (folder name = --folder / --graph_path)
data/fusion360/seg/*.seg                 # Fusion360 segmentation labels
```

Pipeline per STEP file (`preprocess.py: process_single_file → build_graph`):
1. Load with occwl `Compound`; take the single solid (or wrap compound). **Scale to unit box** `[-1,1]³` centered at origin (`scale_solid_to_unit_box`).
2. Optionally apply deterministic random rotation (seeded by file-path hash) for `test_rotated`.
3. Build face-adjacency graph (`occwl.graph.face_adjacency`).
4. Per face (`geometry_features.process_single_face`): sample 10×10 UV grid (points, normals, visibility mask), compute LRF (`compute_local_frame`, handles `TopAbs_REVERSED` orientation by flipping N), produce global `x` and local `x_local` grids, extract face attributes, cast hemisphere rays (max distance = 2× bbox diagonal) → 6-channel vision grid.
5. Segmentation labels are parsed from the **name string of `ADVANCED_FACE` entities in the STEP file itself** (`extract_step_face_labels`); missing → label −1.
6. Save as `.bin` via `dgl.save_graphs`. **By default compressed to float16** (`--no_compress` disables); tensors are cast back to float32 at load time in `BaseDataset._to_float32`.

Two ray-casting backends (`ray_casting.py`):
- **B-rep exact** (`raycast_hemisphere`): OCC `IntCurvesFace_ShapeIntersector`, per-ray Python loop, hit-normal via surface projection. Slow but exact.
- **Mesh** (`MeshRayCaster` + `raycast_hemisphere_mesh`): tessellate once (`BRepMesh_IncrementalMesh`), batch-intersect with trimesh (Embree via `embreex` if available, else pure-Python fallback that is orders of magnitude slower). Ray origins are offset by `eps = max(2·linear_deflection, 1e-3)` along the ray to avoid self-hits; `eps` is added back to distances.

Hemisphere sampling (`hemisphere_grid_sampling`): elevation bin centers at `(i+0.5)/n_el · 90°`, azimuth at `j/n_az · 360°`; directions built in the LRF axes; the special case 1×1 = single ray along `N`.

## 5. Graph schema (node/edge features in `.bin` files)

| Key | Shape | Content |
|---|---|---|
| `ndata["x"]` | (N, 10, 10, 7) | Global-frame UV grid: xyz (0–2), normal (3–5), trim mask (6) |
| `ndata["x_local"]` | (N, 10, 10, 7) | Same, in the per-face LRF (rotation-invariant) |
| `ndata["vision_grids"]` | (N, el, az, 6) | ch 0–2: **OV** occupancy/distance/dot; ch 3–5: **IV** occupancy/distance/dot |
| `ndata["face_feat"]` | (N, 7) | `[Plane, Cylinder, Cone, Sphere, Torus, FaceArea, RationalNurbs]` — **area is index 5**, not last |
| `ndata["vision_features"]` | (N, 4) | Allocated but always zeros; dropped in `forward()` — dead |
| `ndata["y"]` | (N,) int64 | Per-face segmentation label (segmentation datasets only) |
| `edata["x"]` / `edata["edge_feat"]` | (E, 10, 12) / (E, 10) | Edge UV-grids / AAG edge attributes — only with `--edge_info`; **unused by the model** (dropped in `forward()`) |

Graphs are stored **channels-last**; `FOVNet.forward` permutes `x`/`x_local`/`vision_grids` to NCHW in place and pops all unneeded `ndata`/`edata` keys to save memory.

## 6. Model architecture (code-level)

`fovnet/FOVNet.py` + `fovnet/encoders.py`:

```
per face:
  x_local (7,10,10) ──► SurfaceEncoder: 3× [Conv3x3→BN→LeakyReLU] (32→64→128)
                        → AdaptiveAvgPool → FC → srf_emb_dim (default 128)
  OV grid (3,el,az) ──► VisionGridEncoder ┐  2× [circular-pad(az) + zero-pad(el) → Conv3x3→BN→ReLU]
  IV grid (3,el,az) ──► VisionGridEncoder ┘  (32→64) → AvgPool → FC → vision_emb_dim (default 128)
                        (+2 extra input channels: normalized el/az coordinate maps, CoordConv-style)
  face_feat (7,)    ──► passed through raw
concat ─► combination_fc: FC 256 → FC input_graph_dim (64)      # fusion MLP
      ─► GraphEncoder: 3× GATConv (hidden 64, 4 heads, residual, ELU;
                       last layer 1 head → graph_emb_dim (128), no residual)
         → node_emb (N,128) and graph_emb = MaxPool over nodes (B,128)
heads (_MLPHead: 256→128→num_classes, BN+Dropout):
  classification: clf(graph_emb)
  segmentation:   seg([node_emb ‖ graph_emb repeated per node])   # 2×graph_emb_dim input
```

- OV and IV have **separate encoder instances** (no weight sharing); channel selection via `ov_channels=(0,1,2)`, `iv_channels=(3,4,5)` slicing of `vision_grids`.
- Circular padding is applied **only along azimuth** (periodic 360°); elevation gets zero padding.
- `FOVNetModule` (Lightning): cross-entropy loss, torchmetrics Accuracy (+ JaccardIndex for segmentation), AdamW, `save_hyperparameters()` so checkpoints restore the full config.
- Ablation switches (`vision`, `use_ov`, `use_iv`, `use_uv`, `use_face_feat`, `local_uv`) simply omit encoders/streams; the fusion MLP input dim adapts.

## 7. Datasets

| Registry key (`main.py`) | Class | Task | Classes | Notes |
|---|---|---|---|---|
| `solidletters` | `SolidLetters` | classification | 26 | Label = first char of filename; ~60 invalid fonts filtered at preprocess time (`SOLIDLETTERS_INVALID_FONTS`) |
| `traceparts` | `TraceParts` | classification | 6 | Stratified 80/10/10 split |
| `fusion360` | `Fusion360` | segmentation | 8 | Labels from `data/fusion360/seg/*.seg` |
| `mfcad++` | `MFCAD` | segmentation | 25 | Labels embedded in STEP `ADVANCED_FACE` names |

- `SEGMENTATION_DATASETS = {"fusion360", "mfcad++"}` — segmentation mode is inferred from the dataset name, not a flag.
- **`datasets/bendfm.py` (BenDFM, 2-class) exists but is not wired into `DATASET_REGISTRY`** — add it there to use it. The `REGRESSION` cfg flag is plumbed through `main.py` but hardcoded `False`.
- `BaseDataset` loads `.bin` lazily per item; a corrupt file yields an empty graph instead of killing the epoch. `--aug` applies a random rotation to features at load time (`util.rotate_face_features`) — note this rotates the *stored* features, meaningful only for global-frame features; LRF/FoV features are invariant anyway.

## 8. Gotchas & code-vs-paper discrepancies

Things an agent should know before editing or drawing conclusions from the paper alone:

1. **Embedding dims differ from the paper.** Paper Fig. 6/§3.3 says each encoder projects to 64-D and fusion yields a 64-D node embedding. Code defaults: `srf_emb_dim=128`, `vision_emb_dim=128`, fusion output `input_graph_dim=64`, GAT output `graph_emb_dim=128`. The fusion output matches (64); the per-encoder dims do not.
2. **Optimizer**: paper says Adam; code uses **AdamW**. Paper patience 30; code default `--patience 25`.
3. **Segmentation head**: paper says node embeddings go "directly" to the prediction head; code concatenates the pooled **graph embedding** to every node embedding first.
4. **`face_feat` ordering**: paper describes "6D one-hot surface type + area"; code order is 5 type flags, then **area at index 5**, then `RationalNurbs` at index 6. Only `face_feat[:, :7]` is consumed.
5. **Half-precision storage**: preprocessing compresses graphs to fp16 **by default** (`--no_compress` opts out); everything is upcast to fp32 at load. The only affected channel is **face area** (`face_feat[:, 5]`): fp16 flushes areas < ~3e-8 (unit-box scale) to exactly 0 and degrades precision below 6.1e-5. Measured on the full Fusion360 train+val (36,458 models, 803k faces, real user CAD): **89 faces flushed to 0** (0.011% of faces, 39 models = 0.11%; true areas 4e-12–2.7e-8, i.e. degenerate slivers), and ~2% of faces sit in the subnormal band with median rel. error 0.05% (p99 ~14%). On the clean synthetic benchmarks (SolidLetters/MFCAD++ smoke samples) nothing came near the range (min area 0.34, worst error 0.04%). Practical training impact is negligible — `face_feat` feeds Linear+BatchNorm, which can't distinguish 1e-8 from 0 on an O(1)-scale column anyway. Distances can't hit this: ray tolerance (1e-4) / mesh-caster epsilon (≥1e-3) bound them well above fp16's floor, and coordinates/normals/dots are O(1) signed values where near-zero flushing is harmless.
6. **`test_rotated` is a preprocessing product**, not an on-the-fly transform: run `preprocess.py --rotate` to generate `data/<ds>/<split>_rotated/<folder>/*.bin`. Training always tries to build a `test_rotated` dataloader — it must exist on disk.
7. **`preprocess.py --all` defaults to `True`** (argparse `action="store_true", default=True`), so `--split` alone does *not* limit processing — you cannot disable `--all` from the CLI without editing the default. Same pattern for `--seg`.
8. **Rotation in preprocessing is deterministic** per file (`hash(path) % 2^32` seeds three Euler angles) — reproducible rotated test sets.
9. **CoordConv channels**: `VisionGridEncoder` appends normalized elevation/azimuth coordinate maps (2 extra input channels) — not mentioned in the paper.
10. **Edge features are extracted but never used** by the model (`forward` clears `edata`); the paper deliberately omits edge descriptors (UV-Net ablations showed minimal gain).
11. **`vision_features` (N,4) is dead** — allocated as zeros in `build_graph`, dropped in `forward`.
12. **Mesh vs B-rep rays**: `--mesh_rays` is much faster but approximate (tessellation deflection = bbox-max-extent/100; self-hit epsilon offset). Without `embreex`, trimesh silently falls back to a very slow pure-Python intersector (a warning is logged).
13. **occwl comes from the custom `lambouj` conda channel**; `pythonocc-core` needs `libgl` even headless. Use pixi, not bare pip.
14. **Determinism**: `main.py` sets `torch.use_deterministic_algorithms(True, warn_only=True)` and `CUBLAS_WORKSPACE_CONFIG`; keep new ops deterministic-safe or they will emit warnings.
15. **Known limitations (paper §5)**: single-part B-reps of moderate complexity only; equiangular hemisphere mapping has polar distortion (spherical CNNs suggested); UV reparameterizations (axis flips/swaps) are not handled; ray casting is CPU-bound (PythonOCC).

## 9. Conventions

- Python 3.9 compatible syntax; built-in generics (`list`, `tuple`) in type hints; Ruff for linting (`ruff.toml`), `pixi run lint-fix` to autofix.
- Comment style uses `# ── Section ──` dividers; keep exception handling specific (recent refactors narrowed bare `except`).
- File/branch: git repo on `main`; commit style is short imperative summaries.
