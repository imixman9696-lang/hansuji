import numpy as np
from PIL import Image
from scipy import ndimage

src = Image.open("main.png").convert("RGBA")
arr_full = np.array(src)
H, W = arr_full.shape[:2]

ink = (arr_full[:, :, 0] < 248) | (arr_full[:, :, 1] < 248) | (arr_full[:, :, 2] < 248)
labels, n = ndimage.label(ink, structure=np.ones((3, 3), dtype=int))
objects = ndimage.find_objects(labels)

# sprite canvas rectangles (also used verbatim in index.html)
boxes = {
    "fish-a": (800, 260, 1180, 620),
    "fish-b": (1750, 345, 1970, 565),
    "fish-c": (2280, 400, 2580, 700),
    "fish-d": (430, 500, 980, 853),
    "fish-e": (1000, 730, 1230, 900),
    "fish-f": (2155, 790, 2335, 950),
    "fish-g": (270, 955, 550, 1235),
    "fish-h": (2740, 1020, 2920, 1160),
    "fish-i": (280, 1790, 520, 1960),
    "kelp-left": (480, 600, 1020, 1700),
    "kelp-right": (2330, 680, 2820, 1700),
}

# a point on each fish body; the connected ink component under (or near) the
# seed is the fish outline, and every component nested in its bbox (eyes,
# stripes) is pulled in with it
fish_seeds = {
    "fish-a": [(995, 440), (870, 590)],
    "fish-b": [(1860, 455)],
    "fish-c": [(2430, 550)],
    "fish-d": [(820, 585), (545, 650), (505, 695), (615, 700), (555, 743), (865, 747), (905, 778)],
    "fish-e": [(1080, 865)],
    "fish-f": [(2240, 925)],
    "fish-g": [(410, 1095)],
    "fish-h": [(2835, 1095)],
    "fish-i": [(395, 1835)],
}

# base clutter (rocks/coral/urchins) to keep out of the kelp sprites
kelp_exclusion_zones = {
    "kelp-left": [
        (470, 1280, 660, 1700),
        (660, 1570, 890, 1700),
        (860, 1330, 1030, 1700),
    ],
    "kelp-right": [
        (2270, 1230, 2430, 1700),
        (2420, 1370, 2710, 1700),
        (2670, 1440, 2830, 1700),
    ],
}


def label_at(x, y, r_max=30):
    if ink[y, x]:
        return labels[y, x]
    for r in range(1, r_max + 1):
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        win = labels[y0:y1, x0:x1]
        nz = win[win > 0]
        if nz.size:
            return nz[0]
    return 0


def comp_bbox(lab):
    sl = objects[lab - 1]
    return (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)  # x0,y0,x1,y1


# fish masks: a component belongs to a fish when it fits inside the fish's
# canvas box AND touches the padded bbox of a seed component. The seed may hit
# a small detail (gill mark), so the pad lets the full outline join via
# intersection rather than containment.
PAD = 20
fish_labels = {}          # sprite name -> set of labels
all_fish_label_set = set()
comp_bboxes = [comp_bbox(l) for l in range(1, n + 1)]

for name, seeds in fish_seeds.items():
    box = boxes[name]
    cores = set()
    for (x, y) in seeds:
        lab = label_at(x, y)
        if lab:
            cores.add(lab)
    regions = []
    for lab in cores:
        bx0, by0, bx1, by1 = comp_bbox(lab)
        regions.append((bx0 - PAD, by0 - PAD, bx1 + PAD, by1 + PAD))
    members = set(cores)
    changed = True
    while changed:      # grow until stable so chains of touching parts join
        changed = False
        for lab in range(1, n + 1):
            if lab in members:
                continue
            cx0, cy0, cx1, cy1 = comp_bboxes[lab - 1]
            if not (cx0 >= box[0] and cy0 >= box[1] and cx1 <= box[2] and cy1 <= box[3]):
                continue
            for (rx0, ry0, rx1, ry1) in regions:
                if cx0 < rx1 and cx1 > rx0 and cy0 < ry1 and cy1 > ry0:
                    members.add(lab)
                    regions.append((cx0 - PAD, cy0 - PAD, cx1 + PAD, cy1 + PAD))
                    changed = True
                    break
    fish_labels[name] = members
    all_fish_label_set |= members
    print(name, "components:", len(members))

fish_pixel_mask = np.isin(labels, list(all_fish_label_set))

sprite_masks = {}
for name, box in boxes.items():
    x0, y0, x1, y1 = box
    local_ink = ink[y0:y1, x0:x1]
    if name == "fish-i":
        # flatfish outline is fused with the sand line into one component, so
        # component ownership can't isolate it — take all ink in the box and
        # animate it with the seam-friendly boil filter instead of a wiggle
        local = local_ink.copy()
    elif name.startswith("fish"):
        local = np.isin(labels[y0:y1, x0:x1], list(fish_labels[name]))
    else:
        local = local_ink & ~fish_pixel_mask[y0:y1, x0:x1]
        for (zx0, zy0, zx1, zy1) in kelp_exclusion_zones.get(name, []):
            ix0, iy0 = max(x0, zx0), max(y0, zy0)
            ix1, iy1 = min(x1, zx1), min(y1, zy1)
            if ix0 < ix1 and iy0 < iy1:
                local[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = False
    sprite_masks[name] = local

    out = arr_full[y0:y1, x0:x1].copy()
    out[:, :, 3] = np.where(local, 255, 0)
    Image.fromarray(out).save(f"sprites/{name}.png")

# stone-text swap: the 海友 text (main.png) fades into HAEWOO (main2.png).
# Only pixels that differ between the two versions belong to the text, so the
# stone speckles around it stay in the background untouched.
arr2_full = np.array(Image.open("main2.png").convert("RGBA"))
TEXT_BOX = (1420, 655, 1740, 860)
tx0, ty0, tx1, ty1 = TEXT_BOX
a = arr_full[ty0:ty1, tx0:tx1]
b = arr2_full[ty0:ty1, tx0:tx1]
diff = np.abs(a[:, :, :3].astype(int) - b[:, :, :3].astype(int)).sum(axis=2) > 30
diff = ndimage.binary_dilation(diff, iterations=2)
ink_a = (a[:, :, 0] < 248) | (a[:, :, 1] < 248) | (a[:, :, 2] < 248)
ink_b = (b[:, :, 0] < 248) | (b[:, :, 1] < 248) | (b[:, :, 2] < 248)

hanja = a.copy()
hanja[:, :, 3] = np.where(diff & ink_a, 255, 0)
Image.fromarray(hanja).save("sprites/text-hanja.png")

english = b.copy()
english[:, :, 3] = np.where(diff & ink_b, 255, 0)
Image.fromarray(english).save("sprites/text-en.png")

# scene background: original with every sprite's ink painted white
scene = arr_full.copy()
for name, box in boxes.items():
    x0, y0, x1, y1 = box
    m = sprite_masks[name]
    region = scene[y0:y1, x0:x1]
    region[m] = [255, 255, 255, 255]

scene[ty0:ty1, tx0:tx1][diff & ink_a] = [255, 255, 255, 255]

Image.fromarray(scene).save("scene-bg.png")

# bubbles: everything that differs between main.png and main3.png outside the
# text box. Some sit on the black cave interior, so alpha comes from the diff
# itself (not ink thresholds) and RGB from main3.
arr3_full = np.array(Image.open("main3.png").convert("RGBA"))
d3 = np.abs(arr3_full[:, :, :3].astype(int) - arr_full[:, :, :3].astype(int)).sum(axis=2) > 30
d3[ty0:ty1, tx0:tx1] = False          # text swap region is not a bubble
d3_soft = ndimage.binary_dilation(d3, iterations=2)
blob_lab, blob_n = ndimage.label(ndimage.binary_dilation(d3, iterations=6))
blob_sizes = ndimage.sum(d3, blob_lab, range(1, blob_n + 1))

entries = []
for i in range(1, blob_n + 1):
    if blob_sizes[i - 1] < 50:
        continue
    m = (blob_lab == i) & d3_soft
    ys, xs = np.where(m)
    bx0, by0, bx1, by1 = xs.min() - 3, ys.min() - 3, xs.max() + 4, ys.max() + 4
    out = arr3_full[by0:by1, bx0:bx1].copy()
    out[:, :, 3] = np.where(m[by0:by1, bx0:bx1], 255, 0)
    name = f"bubble-{len(entries):02d}"
    Image.fromarray(out).save(f"sprites/{name}.png")
    entries.append((name, bx0, by0, bx1 - bx0, by1 - by0, (by0 + by1) / 2))

# bottom bubbles appear first, as if rising out of the cave
entries.sort(key=lambda e: -e[5])
with open("bubbles.html", "w", encoding="utf-8") as f:
    for rank, (name, x, y, w, h, cy) in enumerate(entries):
        delay = round((1470 - cy) / 1350 * 2.0, 2)
        f.write(f'      <image class="bubble" style="--d:{max(0.0, delay)}s" '
                f'href="sprites/{name}.png" x="{x}" y="{y}" width="{w}" height="{h}"/>\n')
print("bubbles:", len(entries))
print("done")
