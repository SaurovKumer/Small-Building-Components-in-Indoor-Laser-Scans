# Replace the legend of fig_pipeline_real.pdf: keep only what the crop shows.
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject
src, out = sys.argv[1], sys.argv[2]
rd = PdfReader(src); pg = rd.pages[0]
W, H = float(pg.mediabox.width), float(pg.mediabox.height)
plt.rcParams.update({"font.family": "STIXGeneral", "pdf.fonttype": 42})
fig = plt.figure(figsize=(W / 72, H / 72))
fig.patch.set_alpha(0)
# white strip over the old legend (PDF y 3..15 pt)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.add_patch(plt.Rectangle((0, 2.5), W, 13.0, facecolor="white", edgecolor="none"))
hs = [Patch(facecolor="none", edgecolor="#2a78d6", lw=0.6, label="detector box"),
      Patch(facecolor="#e86834", edgecolor="0.5", lw=0.3, label="Switch"),
      Patch(facecolor="#dcd8d4", edgecolor="0.5", lw=0.3, label="Wall")]
fig.legend(handles=hs, loc="center", bbox_to_anchor=(0.5, 9.3 / H), ncol=3, fontsize=6.5,
           frameon=False, handlelength=1.0, columnspacing=0.8)
fig.savefig("overlay.pdf", transparent=True)
ov = PdfReader("overlay.pdf").pages[0]
pg.merge_page(ov)
w = PdfWriter(); w.add_page(pg); w.write(out)
