# ---- FIG-003: data figure, pipeline with real images, review-effort curve (appended to notebook 16) ----
# Uses STN, base3d, hyb, CFG, render_view, detect, segment, merge_dets, features, SET_D, SET_N from the cells above.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
OUT3 = os.path.join(os.path.dirname(OUT), "FIG-003_data_pipeline_review")
os.makedirs(OUT3, exist_ok=True)
INFO3 = {}
SW, OU, DA = 14, 10, 3


def predict_props(train_st, test_st, parts):
    """Per-proposal class probabilities of the 2D classifier (same model and features as train_predict)."""
    train = [it for t in train_st for it in STN[t]["props"] + STN[t]["gtc"]]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=CFG["C"], class_weight="balanced", max_iter=5000, random_state=CFG["seed"]))
    allt = [it for s in test_st for it in STN[s]["props"]]
    Xtr, Xte = features(train, allt, parts)
    clf.fit(Xtr, np.array([it["y"] for it in train]))
    pr = clf.predict_proba(Xte) if len(allt) else np.zeros((0, len(clf.classes_)))
    out, i0 = {}, 0
    for s in test_st:
        n = len(STN[s]["props"]); out[s] = pr[i0:i0 + n]; i0 += n
    return out, list(clf.classes_)


def fused_scores(PD, PN, classes):
    """Damper from D; Switch / Outlet from (1 - w) D + w N, as in the module."""
    w = CFG["fused_w"]; ci = {c: classes.index(c) for c in (DA, OU, SW) if c in classes}
    sc = {c: np.zeros(len(PD)) for c in (DA, OU, SW)}
    if DA in ci: sc[DA] = PD[:, ci[DA]]
    for c in (OU, SW):
        if c in ci: sc[c] = (1 - w) * PD[:, ci[c]] + w * PN[:, ci[c]]
    return sc


def candidates(S, r=None):
    """3D candidates a person would review: proposals whose centroids lie within merge_m are one candidate."""
    r = CFG["merge_m"] if r is None else r
    if not S["props"]: return np.zeros(0, int), []
    C = np.array([S["xyz"][p_["ids"]].mean(0) for p_ in S["props"]])
    comp = cluster_centroids(C, r)
    groups = [np.flatnonzero(comp == k) for k in range(comp.max() + 1)]
    return comp, groups


# ======================= (1) review effort, leave-one-station-out on the seven labelled stations =======================
ALL = [s for s in CFG["train"] + CFG["val"] if s in STN]
KMAX = 150
curves = {c: {"classifier": np.zeros(KMAX + 1), "detector": np.zeros(KMAX + 1)} for c in (SW, OU, DA)}
nobj = {c: 0 for c in (SW, OU, DA)}
per_station = {}
PHR = {SW: ["light switch"], OU: ["electrical outlet", "power socket"], DA: ["smoke detector"]}
for t in ALL:
    S = STN[t]; rest = [s for s in ALL if s != t]
    PD, cls = predict_props(rest, [t], SET_D); PN, cls2 = predict_props(rest, [t], SET_N)
    assert cls == cls2
    sc = fused_scores(PD[t], PN[t], cls)
    comp, groups = candidates(S)
    gd = np.array([p_["gd"] for p_ in S["props"]]) if S["props"] else np.zeros((0, P + 1))
    pts = [np.unique(np.concatenate([S["props"][i]["ids"] for i in g])) for g in groups]
    per_station[t] = dict(proposals=len(S["props"]), candidates=len(groups), seconds=S["seconds"])
    for c in (SW, OU, DA):
        objs = [o for cc, o in S["gobj"] if cc == c]
        nobj[c] += len(objs)
        if not objs or not groups: continue
        # candidate k finds object o if it covers at least 30 % of the object's points (the proposal-recall rule)
        hit = np.array([[np.isin(o, q, assume_unique=False).mean() >= 0.3 for o in objs] for q in pts])
        for name, score in (("classifier", np.array([sc[c][g].max() for g in groups])),
                            ("detector", np.array([gd[g][:, [CFG["phrases"].index(ph) for ph in PHR[c]]].max() for g in groups]))):
            order = np.argsort(-score, kind="stable")
            found = np.zeros(len(objs), bool)
            for K in range(1, KMAX + 1):
                if K <= len(order): found |= hit[order[K - 1]]
                curves[c][name][K] += found.sum()
    print(f"{t}: {len(S['props'])} proposals -> {len(groups)} candidates; objects " +
          ", ".join(f"{CLASS_NAMES[c]} {sum(cc == c for cc, _ in S['gobj'])}" for c in (SW, OU, DA)))

REV = {}
for c in (SW, OU, DA):
    REV[CLASS_NAMES[c]] = dict(objects=nobj[c], **{name: {f"top{K}": int(curves[c][name][K]) for K in (1, 3, 5, 10, 20, 30, 50, 100, 150)}
                                                   for name in ("classifier", "detector")})
    for name in ("classifier", "detector"):
        full = np.flatnonzero(curves[c][name] >= nobj[c]) if nobj[c] else []
        REV[CLASS_NAMES[c]][name]["K_all_found"] = int(full[0]) if len(full) else None
INFO3["review_effort_LOSO"] = REV; INFO3["stations"] = per_station
print("review effort (objects found among the top-K candidates per station, pooled over the 7 held-out stations):")
print(json.dumps(REV, indent=1))

kx = [v[n]["K_all_found"] for v in REV.values() for n in ("classifier", "detector") if v["objects"]]
XMAX = min(KMAX, max([k for k in kx if k is not None] or [KMAX]) + 5) if all(k is not None for k in kx) else KMAX
fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.1), sharey=True)
for a, c in zip(axs, (SW, OU, DA)):
    K = np.arange(0, KMAX + 1)
    for name, st, lab_ in (("classifier", "-", "2D classifier score"), ("detector", "--", "detector phrase score")):
        y = curves[c][name] / max(nobj[c], 1) * 100; y[0] = 0
        a.step(K, y, where="post", ls=st, lw=1.3, color="#eb6834" if name == "classifier" else "0.35", label=lab_)
    a.set_title(f"{CLASS_NAMES[c]} ({nobj[c]} objects)", fontsize=8); a.set_xlabel("candidates reviewed per station")
    a.set_xlim(0, XMAX); a.set_ylim(0, 102); a.grid(lw=0.3, color="0.85")
axs[0].set_ylabel("objects found (%)"); axs[0].legend(loc="lower right", fontsize=6.5, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(OUT3, "fig_review_effort.png"), dpi=250, bbox_inches="tight", pad_inches=0.02); plt.savefig(os.path.join(OUT3, "fig_review_effort.pdf"), bbox_inches="tight", pad_inches=0.02); plt.show(); plt.close(fig)

# ======================= (2) data figure and (3) pipeline with real images: S163, view towards the switch =======================
s = CFG["val"][0] if CFG["val"] else ALL[0]
S = STN[s]
a = np.load(raw_file("Gold", s)); rgb = np.clip(a[:, 3:6], 0, 255).astype(np.uint8); inten = a[:, 6].astype(np.float32); del a
xyz = S["xyz"].astype(np.float64); lab = S["lab"].astype(np.int64)
origin, _ = scanner_position_v4(xyz, S["floor"], CFG["scanner_height_m"])
sw = [o for cc, o in S["gobj"] if cc == SW]
o = max(sw, key=len) if sw else max([o for _, o in S["gobj"]], key=len)
q = xyz[o].mean(0) - origin; q /= np.linalg.norm(q)
vi = int(np.argmax([view_dirs(y_, p_)[0] @ q for y_, p_ in CFG["views"]]))          # the module's own view that looks most directly at it
yw, pt = CFG["views"][vi]
img, I, IDX, proj = render_view(xyz, rgb, inten, origin, yw, pt, size=CFG["size"], fov_deg=CFG["fov"])
g = stretch(I); pics = {"rgb": Image.fromarray(img), "int": Image.fromarray(np.dstack([g, g, g]))}
on = np.isin(IDX, o)
ys, xs = np.nonzero(on)
bx = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else (600, 600, 680, 680)
INFO3["view"] = dict(station=s, view_index=vi, yaw=yw, pitch=pt, object_class=CLASS_NAMES[lab[o[0]]], object_points=int(len(o)),
                     station_points=int(S["n"]), object_pixels=int(on.sum()), bbox=[int(v) for v in bx])

def labimg(IDX, values, lut=LUT):
    out = np.full(IDX.shape + (3,), 255, np.uint8); ok = IDX >= 0; out[ok] = lut[values[IDX[ok]]]; return out

def frame(a):
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_linewidth(0.4); sp.set_color("0.6")

def box(a, b, col="#e34948", lw=1.0):
    a.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=col, lw=lw))

# (2) data figure: colour | intensity | ground truth | PTv3 alone, whole view, the object marked
pad = 18; mark = (bx[0] - pad, bx[1] - pad, bx[2] + pad, bx[3] + pad)
gt_img = labimg(IDX, lab); b3 = base3d.get(s)
panels = [("(a) colour (RGB)", img), ("(b) laser intensity", np.dstack([g, g, g])), ("(c) ground truth", gt_img)]
if b3 is not None: panels.append(("(d) " + CFG["base_name"].split(",")[0] + " alone", labimg(IDX, b3)))
_H = 154.0   # FIG-006: printed width (0.9 x 522 pt), text 8 pt, fixed margins in pt: title 15, legend 22
fig, axs = plt.subplots(1, len(panels), figsize=(0.9 * 522 / 72, _H / 72))
for a, (t_, im) in zip(np.atleast_1d(axs), panels):
    a.imshow(im, interpolation="nearest"); frame(a); a.set_title(t_, fontsize=8); box(a, mark)
_vis = lab[IDX[IDX >= 0]]
present = [c for c in range(18) if (len(_vis) and (_vis == c).mean() > 0.004) or c == lab[o[0]]]
handles = [Patch(facecolor=PAL[c], edgecolor="0.5", lw=0.3, label=CLASS_NAMES[c]) for c in present]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=min(len(handles), 8), fontsize=8, frameon=False, handlelength=1.0, columnspacing=0.8)
plt.subplots_adjust(left=0.004, right=0.996, top=1 - 15 / _H, bottom=22 / _H, wspace=0.025); plt.savefig(os.path.join(OUT3, "fig_data_station.png"), dpi=250); plt.savefig(os.path.join(OUT3, "fig_data_station.pdf"), dpi=300); plt.show(); plt.close(fig)
INFO3["data_fig_classes"] = [CLASS_NAMES[c] for c in present]

# (3) pipeline with real images, cropped around the object: boxes | SAM masks | classified 3D proposals | final labels
raw_d = [d for src in CFG["sources"] for d in detect(pics[src])]
dets = merge_dets(raw_d) if raw_d else []
masks = segment(pics["rgb"], [d[0] for d in dets]) if dets else []
PDs, cls = predict_props(CFG["train"], [s], SET_D); PNs, _ = predict_props(CFG["train"], [s], SET_N)
sc = fused_scores(PDs[s], PNs[s], cls)
ci = {c: i for i, c in enumerate(cls)}
pv = np.full(S["n"], -1, np.int64)                          # class of the proposals of this view, per point
best = np.zeros(S["n"])
nprop_view = 0
for j, p_ in enumerate(S["props"]):
    if p_["view"] != vi: continue
    nprop_view += 1
    probs = {c: sc[c][j] for c in (DA, OU, SW)}
    pbg = PDs[s][j, ci[-1]] if -1 in ci else 0.0
    c_best = max(probs, key=probs.get)
    c_ = c_best if probs[c_best] >= pbg else 17               # background proposals drawn in the clutter grey
    upd = probs[c_best] > best[p_["ids"]]
    pv[p_["ids"][upd]] = c_; best[p_["ids"][upd]] = probs[c_best]
c0x, c0y = (bx[0] + bx[2]) // 2, (bx[1] + bx[3]) // 2; H = 170
cr = (max(c0x - H, 0), max(c0y - H, 0), min(c0x + H, CFG["size"]), min(c0y + H, CFG["size"]))
sl = (slice(cr[1], cr[3]), slice(cr[0], cr[2]))
in_crop = lambda b: b[2] > cr[0] and b[0] < cr[2] and b[3] > cr[1] and b[1] < cr[3]
ov = img.astype(float).copy(); rs = np.random.RandomState(1)
for m in masks:
    col = rs.randint(40, 230, 3); ov[m] = 0.45 * ov[m] + 0.55 * col
prop_img = img.copy(); okp = (IDX >= 0); okp[okp] = pv[IDX[okp]] >= 0
prop_img = (0.35 * prop_img + 0.65 * 255).astype(np.uint8); prop_img[okp] = LUT[pv[IDX[okp]]]
fin = hyb[s] if (isinstance(hyb, dict) and s in hyb) else (b3 if b3 is not None else lab)
fin_img = labimg(IDX, fin)
pp = [("(a) detector boxes", img, True), ("(b) SAM masks", ov.astype(np.uint8), False),
      ("(c) proposals in 3D,\nclassified", prop_img, False), ("(d) after voting\nand size check", fin_img, False)]
_H = 163.0   # FIG-006: printed width (0.9 x 522 pt), text 8 pt, fixed margins in pt: two-line title 24, legend 22
fig, axs = plt.subplots(1, 4, figsize=(0.9 * 522 / 72, _H / 72))
for a, (t_, im, draw) in zip(axs, pp):
    a.imshow(im[sl], interpolation="nearest"); frame(a); a.set_title(t_, fontsize=8)
    if draw:
        for d in dets:
            b = d[0]
            if in_crop(b): box(a, (b[0] - cr[0], b[1] - cr[1], b[2] - cr[0], b[3] - cr[1]), "#2a78d6", 0.6)
# legend: only the classes that actually appear in panels (c) and (d) of the crop, plus the box colour of (a)
_pc = pv[IDX[sl][IDX[sl] >= 0]]; _fc = fin[IDX[sl][IDX[sl] >= 0]]
used = [c for c in [SW, OU, DA, 17] + sorted(set(range(18)) - {SW, OU, DA, 17}) if (_pc == c).mean() > 0.002 or (_fc == c).mean() > 0.002]
handles = [Patch(facecolor="none", edgecolor="#2a78d6", lw=0.6, label="detector box")] + [Patch(facecolor=PAL[c], edgecolor="0.5", lw=0.3, label=(CLASS_NAMES[c] if c != 17 else "background / clutter")) for c in used]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), fontsize=8, frameon=False, handlelength=1.0, columnspacing=0.8)
plt.subplots_adjust(left=0.004, right=0.996, top=1 - 24 / _H, bottom=22 / _H, wspace=0.025); plt.savefig(os.path.join(OUT3, "fig_pipeline_real.png"), dpi=250); plt.savefig(os.path.join(OUT3, "fig_pipeline_real.pdf"), dpi=300); plt.show(); plt.close(fig)
INFO3["pipeline_fig"] = dict(view_detections_merged=len(dets), boxes_in_crop=int(sum(in_crop(d[0]) for d in dets)),
                             proposals_from_view=nprop_view, crop=[int(v) for v in cr],
                             object_recall_final=round(float((fin[o] == lab[o[0]]).mean()), 3),
                             object_recall_3d=round(float((b3[o] == lab[o[0]]).mean()), 3) if b3 is not None else None)
print("FIG-003:", json.dumps({k: v for k, v in INFO3.items() if k != "review_effort_LOSO"}))
with open(os.path.join(OUT3, "summary_fig003.json"), "w") as fh: json.dump(INFO3, fh, indent=1)
print("saved:", sorted(os.listdir(OUT3)))
