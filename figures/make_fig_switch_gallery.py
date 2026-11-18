"""Figure 4 (switch gallery) from the executed DIAG-001 notebook (notebooks/executed/12_diag_v2_DIAG-001.ipynb).
The galleries are the inline images of cell 10 (order: S163, S169, S170, S171, S180); the Switch row is the last row.
Crops used (column index in the Switch row, Switch probability printed in the notebook title):
  S163 (val, classifier trained on the 5 training stations): 0 (0.99), 3 (0.92), 7 (0.65), 11 (0.55)
  S171 (test, classifier trained on all 7 labelled stations): 3 (0.19), 5 (0.18), 10 (0.12), 11 (0.11)  -- cream wall plates, h 1.14 m
Run from the repository root: python paper/figures/make_fig_switch_gallery.py"""
import json, base64, io
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

nb = json.load(open("notebooks/executed/12_diag_v2_DIAG-001.ipynb", encoding="utf-8"))
imgs = [base64.b64decode(o["data"]["image/png"]) for o in nb["cells"][10].get("outputs", []) if "image/png" in o.get("data", {})]
gal = {s: np.array(Image.open(io.BytesIO(b)).convert("RGB")).astype(int) for s, b in zip(["S163", "S169", "S170", "S171", "S180"], imgs)}

def switch_crops(im):
    band = im[945:1142]; nw = np.abs(band - 255).sum(2) > 30; col = nw.any(0); segs, s = [], None
    for x, v in enumerate(col):
        if v and s is None: s = x
        if not v and s is not None:
            if x - s > 40: segs.append((s, x))
            s = None
    out = []
    for a, b in segs:
        ys = np.where(nw[:, a:b].any(1))[0]; out.append(Image.fromarray(band[ys[0]:ys[-1] + 1, a:b].astype("uint8")))
    return out

# FIG-005: drawn at the column width of the two-column layout (252 pt) with all text at 8 pt (= caption size),
# so it reads at 8 pt there (included with width=\linewidth; in the wider review layout it is scaled up).
FIG_W, FS = 252 / 72, 8
plt.rcParams.update({"font.family": "STIXGeneral", "font.size": FS, "pdf.fonttype": 42, "mathtext.fontset": "stix"})
rows = [("S163", [(0, 0.99), (3, 0.92), (7, 0.65), (11, 0.55)]), ("S171", [(3, 0.19), (5, 0.18), (10, 0.12), (11, 0.11)])]
fig, ax = plt.subplots(2, 4, figsize=(FIG_W, 2.12))
for r, (st, items) in enumerate(rows):
    crops = switch_crops(gal[st])
    for k, (i, p) in enumerate(items):
        im = crops[i]; w, h = im.size; s = max(w, h); sq = Image.new("RGB", (s, s), (255, 255, 255)); sq.paste(im, ((s - w) // 2, (s - h) // 2))
        a = ax[r, k]; a.imshow(sq, interpolation="lanczos"); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_linewidth(0.4); sp.set_color("0.6")
        a.set_xlabel(f"$p={p:.2f}$", fontsize=FS, labelpad=1.5)
fig.text(0.045, 0.765, "(a) Val. S163:\nlabelled switch", rotation=90, va="center", ha="center", fontsize=FS)
fig.text(0.045, 0.285, "(b) Test S171:\nlook-alike plates", rotation=90, va="center", ha="center", fontsize=FS)
plt.subplots_adjust(left=0.115, right=0.997, top=0.997, bottom=0.085, wspace=0.06, hspace=0.2)
plt.savefig("paper/figures/fig_switch_gallery.pdf", dpi=300); plt.savefig("paper/figures/fig_switch_gallery.png", dpi=150)
