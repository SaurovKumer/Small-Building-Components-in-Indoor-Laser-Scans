# Small building components in indoor laser scans: a scanner-view 2D module on top of 3D segmentation networks

Code and result files for the paper *"Small Building Components in Indoor Laser Scans: A Scanner-View 2D Module on Top of 3D Segmentation Networks"* (submitted to *Automation in Construction*).

The method adds small-component labels (smoke detectors, outlets, switches) to the output of any 3D semantic segmentation network for terrestrial laser scanning (TLS) stations. It needs no extra photos:

1. **Scanner position from the point order.** Consecutive points of an exported TLS station form vertical scan lines; the planes of these lines meet at the scanner. A least-squares intersection gives the scanner position without labels (1.6–4.2 cm from the tripod footprint where it is visible).
2. **Rendering.** 25 pinhole views (RGB and laser intensity) are rendered from that position.
3. **Proposals.** Grounding DINO proposes boxes, SAM turns them into masks, and the masks are lifted to the visible 3D points.
4. **Classification and voting.** A light classifier (detector scores, 3D context, DINOv2 embeddings) labels each proposal; proposals vote per point.
5. **Combination.** The 3D label of a point becomes Damper, Outlet or Switch only where at least three proposals agree and the relabelled segment is not larger than 1.5 times the largest training object of that class.

Everything was run on Kaggle notebooks with an NVIDIA T4 GPU.

## Main results (3DSES Gold)

Validation (stations S163 and S169, every original point, mIoU %; PTv3 mean ± s.d. over three seeds):

| Method | mIoU | Small-component mIoU |
|---|---|---|
| PointNet++ B1 (RGB + intensity), 3 seeds | 58.39 ± 0.58 | 33.81 ± 0.44 |
| PTv3, 3 seeds | 68.77 ± 1.02 | 50.54 ± 1.63 |
| PTv3 + 2D module, 3 seeds | 75.91 ± 1.03 | 71.95 ± 1.63 |

Hidden test stations (S170, S171, S180; scored by the benchmark server, Codabench competition 6927, task "Gold – Real labels"):

| Method | mIoU (seed 0 / 1 / 2) | Mean ± s.d. |
|---|---|---|
| PTv3 | 64.85 / 60.21 / 62.40 | 62.49 ± 2.32 |
| PTv3 + 2D module | 65.45 / 60.99 / 63.31 | 63.25 ± 2.23 |
| Best published result (Swin3D-L + intensity) | — | 52.02 |

On test the module improves every seed (+0.6 to +0.9 mIoU); the gain comes from smoke detectors (Damper IoU +12 to +18 points). Switches and outlets whose design does not occur in the training stations are not found (see the paper and notebook 12).

## Data

3DSES (Mérizette et al., VISAPP 2025). The notebooks download the Gold variant and the test stations from the Hugging Face mirror `heig-vd-geo/3DSES` with `huggingface_hub`. The original release is on Zenodo: https://zenodo.org/records/13323342. Check the dataset licence before redistributing any of it. Split used throughout: training S164, S165, S168, S178, S179; validation S163, S169; test S170, S171, S180 (labels hidden).

## Requirements

The notebooks are written for Kaggle (GPU T4, Internet on) and install what they need themselves:

- PyTorch (pre-installed on Kaggle; notebook 14 checks for `torch >= 2.10`)
- `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `Pillow`, `huggingface_hub`
- `transformers` (Grounding DINO `IDEA-Research/grounding-dino-base`, SAM `facebook/sam-vit-base`, DINOv2 `facebook/dinov2-base`)
- PointNet++: the OpenPoints code of PointNeXt (`github.com/guochengqian/PointNeXt`, cloned by notebook 04)
- PTv3: `point-transformer-v3==0.1.2` (pip, plain PyTorch, no spconv)

`requirements.txt` lists the Python packages for a local run; install PyTorch for your CUDA version first.

## Notebooks (run in this order)

| Notebook | GPU | What it does | Paper |
|---|---|---|---|
| `03_preprocess_gold.ipynb` | no | Downloads 3DSES Gold, checks the class mapping, voxelises the stations at 2 cm and 1 cm | Sec. 3, Table 2 |
| `04_baseline_pointnet2.ipynb` | yes | PointNet++ (OpenPoints) B0 = RGB, B1 = RGB + intensity; evaluates on every original point and writes test labels | Table 4 |
| `05_fullres_validation.ipynb` | yes | Optional: re-scores saved checkpoints on every original point and computes the voxel oracle | Sec. 5.1 |
| `11_scanner_position.ipynb` | no | Label-free scanner position from the scan lines, checked against the tripod footprint | Sec. 4.2, Fig. 3 |
| `13_loso_2d_module.ipynb` | yes | Leave-one-station-out evaluation of the 2D module (feature sets D and N, D/N fusion) | Table 7 |
| `10_hybrid_pointnet2_2d.ipynb` | yes | PointNet++ B1 + 2D module (validation protocol and test submission zips) | Table 6 |
| `14_ptv3_baseline.ipynb` | yes | PTv3 under the same protocol as notebook 04 | Tables 4, 6 |
| `15_hybrid_ptv3.ipynb` | yes | PTv3 + 2D module; validation-selected setting and test submission zips | Tables 4, 5, 6 |
| `12_diag_test_small_objects.ipynb` | yes | Diagnostics on the test stations: proposal funnel and galleries of the best-ranked proposals | Sec. 5.4, Fig. 6 |
| `16_paper_figures.ipynb` | yes | Paper figures: scanner position, qualitative close-ups, data view, pipeline on real images, review effort | Figs. 2–5, 7 |

Each notebook starts with a `CFG` dictionary. The paper runs used these settings:

| Run | Notebook | Settings |
|---|---|---|
| EXP-001 / 004 / 006 (B0, seeds 0 / 1 / 2) | 04 | `use_intensity=False`, `seed=0/1/2` |
| EXP-002 / 003 / 005 (B1, seeds 0 / 1 / 2) | 04 | `use_intensity=True`, `seed=0/1/2` |
| EXP-007 / 008 / 009 (PTv3, seeds 0 / 1 / 2) | 14 | `seed=0/1/2`, `class_weight=None` |
| EXP-011 (PTv3, class-balanced loss) | 14 | `seed=0`, `class_weight="cb"`, `cb_beta=0.9999` |
| HYB-004 (PointNet++ + 2D) | 10 | input: output of notebook 04, run EXP-003 |
| HYB-005 / 006 / 007 (PTv3 + 2D) | 15 | `base_run` = EXP-007 / 008 / 009; input: output of notebook 14 for that run |

On Kaggle a notebook attached as input always resolves to its newest version. To use the output of an older run, save that output as a Kaggle dataset and attach the dataset instead (notebooks 10 and 15 search `/kaggle/input` recursively for the folder named by `base_run`).

### Test submissions

Notebooks 10 and 15 write zip files with the layout the benchmark expects (`real_labels/gold/S170labels.npy`, `S171labels.npy`, `S180labels.npy`; one `uint8` class ID per original point in file order). Upload them by hand on Codabench (competition 6927, "Gold – Real labels"). The benchmark allows 4 submissions per day and 50 in total.

## Figure scripts (`figures/`)

- `fig_pipeline.tex` — TikZ source of the method overview (Fig. 1).
- `make_fig_switch_gallery.py` — Fig. 6 from the galleries saved by notebook 12.
- `make_fig_review_effort.py` — Fig. 7 from `results/FIG-003_data_pipeline_review/review_curves_FIG003b.json`.
- `fig003_cell.py` — the figure cell of notebook 16 (data view, pipeline on real images, review effort).
- `fix_pipeline_real_legend.py` — replaces the legend of `fig_pipeline_real.pdf` with the classes shown in the crop.

## Result files (`results/`)

One folder per run, named by its ID (EXP = 3D training, HYB = hybrid, LOSO = leave-one-station-out, CODA = benchmark submission, SUM = summary over seeds, FEAS / EVAL / DIAG / FIG = feasibility, evaluation, diagnostics and figure runs). They contain only small JSON files: final metrics, training logs, confusion rows and per-station counts. Checkpoints, predictions and submission zips are not included. Useful entry points:

- `SUM-001_baselines_3seeds/summary.json` — PointNet++ B0 / B1 over three seeds
- `SUM-004_hybrid_ptv3_3seeds/summary.json` — PTv3 and PTv3 + 2D on validation over three seeds
- `SUM-005_test_ptv3_3seeds/summary.json` — the six PTv3 test scores and their means
- `LOSO-002_fusion_switch_outlet/summary.json` — leave-one-station-out results of the final 2D module
- `CODA-00x_*/summary.json` — per-class test IoU and accuracy read from the benchmark's scoring logs

## Citation

The paper is under review. Citation details will be added here after publication.

## Licence

The code, notebooks, figure scripts and result files in this repository are released under the MIT License (see `LICENSE`). The 3DSES dataset, the OpenPoints/PointNeXt code, the `point-transformer-v3` package and the pretrained Grounding DINO, SAM and DINOv2 models are not part of this repository; they keep their own licences, which apply when the notebooks download them.

## Acknowledgements

We thank the authors of 3DSES for the dataset and the benchmark server. The experiments were run on GPU time provided by Kaggle.
