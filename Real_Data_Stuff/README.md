# Real Data Pipeline (Prototype)

This folder provides a compact pipeline to process your real faces (not simulated) using a shared OBJ mesh topology and a CSV of per‑face vertex coordinates. It computes interior Gaussian curvatures and provides a quick visual check.

## Use with your real data

Place your real files here:
- `Real_Data_Stuff/Face_OBJs/` — OBJ meshes for your faces. All files must share the same triangle connectivity (same vertex/face indices).
- `Real_Data_Stuff/face_x_70.csv` — each row is a face; columns are flattened `x1,y1,z1,x2,y2,z2,...,xV,yV,zV`.

Then compute curvatures (reproducible options shown):

```powershell
# Fixed random seed, explicit paths if desired
python .\Real_Data_Stuff\run_compute_curvatures.py --seed 42 --csv .\Real_Data_Stuff\face_x_70.csv --objs .\Real_Data_Stuff\Face_OBJs --outputs .\Real_Data_Stuff\outputs
```

What it does:
- Picks one OBJ at random from `Face_OBJs/` to define the triangle faces (topology only).
- Loads each row from `face_x_70.csv`, reshapes into `(V,3)` vertices, and computes angle‑deficit Gaussian curvature per vertex.
- Computes a single interior vertex mask from the topology and, for each face, keeps only interior vertices and normalizes the curvature:
	`exp(curv)` → z‑score → clip to `[-2,2]` → min–max to `[0,1]`.

Outputs (written to `Real_Data_Stuff/outputs/`):
- `interior_mask.npy` — boolean mask for interior vertices (same for all faces)
- `interior_curvatures.npy` — array `(n_faces, n_interior)` with normalized values in `[0,1]`
- `mesh_used.obj` — the OBJ chosen to define the topology for this run

Visualize the result for a given row (default index=0):

```powershell
python .\Real_Data_Stuff\visualize_curvature.py --index 0
```

The visualizer reconstructs the mesh geometry for that row from the CSV, colors only interior vertices by the corresponding row in `interior_curvatures.npy` (viridis colormap), and writes:
- `colored_rowXXX.ply` (with vertex colors; open in MeshLab/CloudCompare)
- `colored_rowXXX.png` (best‑effort render; optional)

### Data format notes
- CSV header is optional — the loader detects column names like `x1,y1,...` and handles both header/no‑header files.
- Column count must match `3 * V` where `V` is the mesh vertex count. Extra columns are truncated; fewer columns raise an error.
- Each CSV row is reshaped to `(V,3)` in the order `[x1,y1,z1, x2,y2,z2, ...]`.

### Assumptions
- All OBJs in `Face_OBJs/` share identical triangle connectivity. The script records the chosen mesh as `mesh_used.obj`.
- PNG rendering is optional (headless environments may skip it). PLY is the reliable artifact for inspection.

## Working with the included fake data (optional)

If you’d like to smoke‑test the pipeline before using your real data, generate a tiny synthetic dataset:

```powershell
python .\Real_Data_Stuff\make_fake_real_data.py --seed 42 --n-obj 3 --n-faces 8
python .\Real_Data_Stuff\run_compute_curvatures.py --seed 42
python .\Real_Data_Stuff\visualize_curvature.py --index 0
```

This creates a few OBJ variants in `Face_OBJs/` and a small `face_x_70.csv` with consistent topology so you can verify outputs quickly.

## Dependencies

Python 3.10+ recommended with: `numpy`, `pandas`, `scipy`, `trimesh`, `matplotlib`.
Your workspace already uses these; install as needed in your active environment.

### Reproducibility notes
- `run_compute_curvatures.py` writes `outputs/metadata.json` capturing seed, file hashes, package versions, and Git commit/branch when available.
- For fully frozen environments, consider exporting pinned requirements via `pip freeze > requirements.txt` and citing the commit SHA used in your runs.

## CurvAE for real-data curvatures (grid search)

We include a compact autoencoder training suite under `Real_Data_Stuff/encoder_files` to learn low-dimensional representations of the real-data interior curvatures.

Files:
- `encoder_files/CurvAE.py` — the autoencoder definition (fully-connected, configurable depths; ELU/LeakyReLU activations).
- `encoder_files/CurvAE_GridSearch.py` — grid search runner with 3-fold CV and deterministic seeding. Outputs per-experiment results as pickles and a consolidated `experiment_metadata.csv`.

Input data: the grid search consumes `Real_Data_Stuff/outputs/interior_curvatures.npy` produced by the curvature pipeline above. Run that first.

Run a quick sanity check (tiny grid):

```powershell
python .\Real_Data_Stuff\encoder_files\CurvAE_GridSearch.py --quick --max-epochs 50
```

Run the full grid (can be compute-intensive):

```powershell
python .\Real_Data_Stuff\encoder_files\CurvAE_GridSearch.py
```

Outputs are written to `Real_Data_Stuff/encoder_files/encoder_data/Curvature/SEED{seed}/experiments/` with a summary CSV at `encoder_data/Curvature/experiment_metadata.csv`.

### Why these hyperparameters?
The choices mirror what performed well in our simulated-data grid searches and early CCA analyses, then adapted slightly for real-data stability:

- Latent dimensions (right-skewed, 10 values): `[10, 15, 20, 30, 40, 60, 80, 120, 200, 400]` — more density at small widths where we consistently saw sharper loss improvements, while still probing mid/high capacity.
- Learning rates: `[1e-4, 3e-4, 1e-3]` — spans conservative to moderately aggressive; 3e-4 often a sweet spot.
- Batch sizes: `[16, 32]` — stable on modest GPUs/CPUs and good for variability reduction on real data.
- Patience: `[5, 8]` — early stopping tuned to avoid overfitting without stalling.
- Activations: `ELU`, `LeakyReLU` — both consistently strong in sim runs; ReLU was omitted due to poorer plateaus.
- Encoder depths: we reuse all architectures we tested for latent dims < 500 in sim experiments (aggressive-to-moderate compression families depending on the latent-width band). Decoders are symmetric.

Each configuration is evaluated with 3-fold CV and the same seed family `{39, 40, 41, 42}` used elsewhere so results are directly comparable.

### Intentional/"arbitrary" choices documented
- Shared topology: we choose a single OBJ from `Face_OBJs/` per run to define triangle connectivity; only interior vertices are used for training (boundary masked out).
- Curvature normalization per face: `exp(curv)` → z-score → clip to `[-2,2]` → min–max to `[0,1]` improves numerical stability and downweights extreme spikes.
- Determinism: fixed split seed (`1`) for train/val/test partition; per-experiment seeds for CV folds and initializations (`{39,40,41,42}`); deterministic PyTorch flags enabled where supported.
- CV setup: 3-fold CV is a measured tradeoff between reliability and runtime for the real dataset sizes.
- Storage layout: results are nested by seed under `encoder_files/encoder_data/Curvature/SEED*/experiments/` to align with our simulated-data outputs and downstream scripts.
