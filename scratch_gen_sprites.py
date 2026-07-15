import numpy as np
from PIL import Image

src = Image.open("main.png").convert("RGBA")

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
    "kelp-left": (480, 660, 1020, 1700),
    "kelp-right": (2330, 680, 2820, 1700),
}

# raw rectangles (not tracked as their own sprites) purely used to keep
# rock/coral/urchin clutter at the base of the kelp out of the kelp sprite
extra_exclusion_zones = {
    "kelp-left": [
        (470, 1280, 660, 1700),   # left coral bush
        (660, 1570, 890, 1700),   # round pinecone coral
        (860, 1330, 1030, 1700),  # right-side coral bush
    ],
    "kelp-right": [
        (2270, 1230, 2430, 1700),  # left coral bush
        (2420, 1370, 2710, 1700),  # rock base
        (2670, 1440, 2830, 1700),  # right coral cluster
    ],
}

# boxes of other tracked objects that spatially overlap a given box and must
# be excluded from it (that sub-area is owned by the other object instead)
exclusions = {
    "kelp-left": ["fish-d", "fish-g", "fish-e"],
    "kelp-right": ["fish-f", "fish-c"],
}


def matte_white(crop):
    arr = np.array(crop)
    white_mask = (arr[:, :, 0] >= 248) & (arr[:, :, 1] >= 248) & (arr[:, :, 2] >= 248)
    arr[white_mask, 3] = 0
    return arr


sprite_arrays = {}
for name, box in boxes.items():
    x0, y0, x1, y1 = box
    crop = src.crop(box).copy()
    arr = matte_white(crop)

    ex_boxes = [boxes[other] for other in exclusions.get(name, [])]
    ex_boxes += extra_exclusion_zones.get(name, [])
    for (ox0, oy0, ox1, oy1) in ex_boxes:
        # intersect with this box, translate to local coords
        ix0, iy0 = max(x0, ox0), max(y0, oy0)
        ix1, iy1 = min(x1, ox1), min(y1, oy1)
        if ix0 < ix1 and iy0 < iy1:
            lx0, ly0, lx1, ly1 = ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0
            arr[ly0:ly1, lx0:lx1, 3] = 0

    sprite_arrays[name] = arr
    Image.fromarray(arr, "RGBA").save(f"sprites/{name}.png")

scene = src.copy()
for name, box in boxes.items():
    x0, y0, x1, y1 = box
    arr = sprite_arrays[name]
    sprite_img = Image.fromarray(arr, "RGBA")
    white_patch = Image.new("RGBA", sprite_img.size, (255, 255, 255, 255))
    mask = sprite_img.split()[3]
    scene.paste(white_patch, (x0, y0), mask=mask)

scene.save("scene-bg.png")
print("done", scene.size)
