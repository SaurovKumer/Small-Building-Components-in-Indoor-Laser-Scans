"""Fig. review effort, redrawn from the curves printed by Kaggle notebook 16 v8 (FIG-003b).
curves[cls][ranking][K] = objects found among the top-K candidates per station, pooled over the
7 held-out stations (K = 0..60)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "STIXGeneral", "font.size": 8, "pdf.fonttype": 42, "mathtext.fontset": "stix"})
C = json.load(open(sys.argv[1]))
XMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 40
# FIG-006: printed width in the two-column layout (0.9 x 522 pt), all text 8 pt, exact width (no tight bbox)
fig, axs = plt.subplots(1, 3, figsize=(0.9 * 522 / 72, 2.15), sharey=True)
for a, name in zip(axs, ("Switch", "Outlet", "Damper")):
    d = C[name]; n = d["objects"]
    for key, st, lab, col in (("classifier", "-", "2D classifier score", "#eb6834"), ("detector", "--", "detector phrase score", "0.35")):
        y = np.array(d[key], float) / n * 100; y[0] = 0
        a.step(np.arange(len(y)), y, where="post", ls=st, lw=1.3, color=col, label=lab)
    a.set_title(f"{name} ({n} objects)", fontsize=8); a.set_xlabel("candidates reviewed per station")
    a.set_xlim(0, XMAX); a.set_ylim(0, 102); a.grid(lw=0.3, color="0.85")
axs[0].set_ylabel("objects found (%)"); axs[0].legend(loc="lower right", fontsize=8, frameon=False)
plt.tight_layout(pad=0.3)
plt.savefig("fig_review_effort.pdf")
plt.savefig("fig_review_effort.png", dpi=250)
