# 3D reconstruction candidates

Phase 0 research for the 3D reconstruction feature, which moved into scope on 2026-10-11 (see
`docs/decisions/004-3d-reconstruction.md`). Compiled on 2026-10-11 by a read-only research agent from primary
sources. Every runtime figure here is **reported or estimated, not measured by us**. Measured numbers will come from
`notebooks/07_3d_smoke_test.ipynb`.

Constraints that decide the choice:
- **Machine:** RTX 5050 8 GB (sm_120), torch 2.14.1+cu130, Python 3.12. No C/C++ compiler and no CUDA toolkit, so
  anything that JIT-compiles CUDA or C++ extensions is blocked unless a matching prebuilt wheel exists.
- **Deploy:** Hugging Face CPU Basic (2 vCPU, 16 GB).
- **Speed:** slow is acceptable for the demo, as long as it really runs.
- **Licences:** non-commercial licences must be flagged in `DATA_SOURCES.md`.

## Feed-forward multi-view models (unposed images to camera poses + depth / point maps)

| Model | Code licence | Weights licence | Installs without a compiler | Notes |
|---|---|---|---|---|
| **Depth Anything 3 (DA3)** | Apache-2.0 ([LICENSE](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/LICENSE)) | Apache-2.0: DA3-SMALL (0.08B), **DA3-BASE (0.12B)**, METRIC-LARGE, MONO-LARGE. **CC BY-NC (flag):** GIANT, NESTED, DA3-LARGE. DA3-LARGE-1.1 has a conflict: the [card](https://huggingface.co/depth-anything/DA3-LARGE-1.1) says Apache, the [README](https://github.com/ByteDance-Seed/Depth-Anything-3#%EF%B8%8F-model-cards) says CC BY-NC, so treat it as NC. | Yes: `depth-anything-3==0.1.1`, a py3-none wheel ([PyPI](https://pypi.org/pypi/depth-anything-3/json)). It pins `numpy<2`. | Exports glb, npz, ply, colmap and gs_ply ([API](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md)). The 3D Gaussian head exists only in GIANT/NESTED (NC) and needs gsplat. |
| MapAnything | Apache-2.0 | `map-anything-apache` Apache-2.0 ([card](https://huggingface.co/facebook/map-anything-apache)); `map-anything` CC BY-NC | Only from git. The `uniception` dependency pulls in torchaudio, so install it with care. | 1.2B parameters; outputs `pts3d`, camera poses and confidence per view ([README](https://github.com/facebookresearch/map-anything)). |
| VGGT | Custom (commercial use allowed except military) | `VGGT-1B` CC BY-NC; `VGGT-1B-Commercial` is gated ([README](https://github.com/facebookresearch/vggt)) | Pure PyTorch; its requirements pin torch 2.3.1, so it would need `--no-deps`. | 1B parameters. |
| π³ / Pi3X | BSD-3 | **CC BY-NC** ([README](https://github.com/yyfz/Pi3#-license)) | Pure PyTorch | Not used (NC). |
| Fast3R | FAIR Noncommercial ([LICENSE](https://github.com/facebookresearch/fast3r/blob/main/LICENSE)) | NC | — | Not used. |
| MASt3R / DUSt3R, CUT3R | CC BY-NC-SA ([mast3r](https://github.com/naver/mast3r/blob/main/LICENSE), [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/LICENSE)) | NC | — | Not used. |
| AnySplat | MIT ([card](https://huggingface.co/lhjiang/anysplat)) | MIT | **Blocked:** needs pytorch3d (git build), torch_scatter and a gsplat cp310 / pt2.2 wheel ([requirements](https://github.com/InternRobotics/AnySplat/blob/main/requirements.txt)) | — |

## Optimisation route (structure-from-motion + Gaussian splatting)

- **pycolmap 4.2.1:** BSD-3, cp312 manylinux wheel ([PyPI](https://pypi.org/pypi/pycolmap/json)), runs on CPU. Sparse SfM on ~40 frames is estimated at 2–8 min (estimate). SIFT struggles on white, glossy paint.
- **gsplat: blocked here.** PyPI only ships 1.5.3 as a py3-none wheel that compiles CUDA on first use (needs nvcc). The [prebuilt index](https://docs.gsplat.studio/whl/gsplat/) stops at cp310 / torch 2.4 / cu124. That also blocks nerfstudio splatfacto and the official DA3 Gaussian install.
- **Brush** ([repo](https://github.com/ArthurBrussee/brush), Apache-2.0): the v0.3.0 Linux binary uses Rust / wgpu on Vulkan, so it needs no CUDA toolkit or compiler. It trains splats from COLMAP-format data and writes `.ply`. Possible GPU-only add-on: DA3 `export_format="colmap"`, then Brush, then a splat `.ply`. It cannot run on the CPU Space.

## Browser viewers (all MIT)

- **three.js 0.186.1:** PLYLoader / GLTFLoader / `THREE.Points` for a coloured point cloud.
- **Spark 2.3.1** ([repo](https://github.com/sparkjsdev/spark)): a three.js splat renderer that loads .ply, compressed .ply, .spz, .splat, .ksplat and .sog. Available from jsdelivr (`@sparkjsdev/spark@2.3.1`). Its `SplatMesh` sits alongside ordinary three.js meshes.
- **GaussianSplats3D:** its README says it is no longer developed and recommends Spark.
- **Damage pins:** one small sphere `Mesh` per pin, a `Raycaster` for clicks, `CSS2DRenderer` labels.

## 2D damage masks to 3D pins

For frame *i*:
1. Resize the damage mask to DA3's processed resolution.
2. Keep pixels whose confidence is above roughly the 40th percentile.
3. Back-project with `X_cam = depth · K⁻¹ [u, v, 1]` and `X_world = Rᵀ (X_cam − t)`, using DA3's world-to-camera extrinsics.
4. The pin is the median of those points.
5. Merge pins from different frames that land within a small radius (DBSCAN).

Write the point cloud from the same prediction, so pins and cloud share one coordinate frame. The GLB export may recentre the scene.

## Choice (to be confirmed by the 3D smoke test)

| Where | First choice | Fallback |
|---|---|---|
| Local GPU | DA3-BASE (Apache-2.0), ~30 frames at `process_res` 504, coloured point cloud `.ply` plus pins JSON in three.js | MapAnything-apache (1.2B; may be tight on 8 GB) |
| CPU Space | DA3-SMALL or DA3-BASE on CPU, ~30 frames at `process_res` 336–504; estimated 1–3 min (not measured). DA3 picks fp16 autocast without bf16 CUDA; on CPU use fp32 or bf16. | pycolmap sparse SfM with pins on the sparse cloud |
| Optional splats | DA3, then COLMAP export, then the Brush binary on the GPU, then Spark | none |

**Input:** only the anonymised frames, so plates and faces never end up in the 3D texture.

**Risks to check in the smoke test:**
- Untested on sm_120 / torch 2.14.
- DA3's `numpy<2` pin against our numpy 2.x; test in an isolated environment first.
- Reflections on white paint create depth artefacts. Mitigate with car-only masking and a confidence threshold.
