"""Generate the wall-photograph fixtures the vision evaluation runs against.

**Why these are drawn rather than downloaded.** A photograph of somebody's wall
is somebody's photograph. CLAUDE.md allows synthetic fixtures in the repository
and is pointedly unsure about redistributing crawled material, and a licensing
question is a poor thing to attach to a fixture that has to ship, run in CI and
be reviewable by a panel. Drawing them also buys the thing a downloaded set
cannot: **declared ground truth**. Nobody has to adjudicate what is "really" in
`brick-exposed-clear.jpg`, because this file put it there.

That honesty cuts both ways and the report says so: these are renderings, not
photographs. They carry the structure a vision model keys on -- bond pattern,
mortar joints, crack geometry, salt bloom, exposure -- and they do not carry
lens blur, real daylight or the mess of a real elevation. What they are fit for
is the measurement this system actually needs, which is **not** "does the model
recognise brick". It is *does the model claim what it cannot see* -- an external
wall from an interior close-up, rising damp from a stain, a structural defect
from a crack. A drawn fixture with a declared "must not claim" list measures
that exactly, and a downloaded one measures it only as well as whoever labelled
it.

Run: `python tools/make_image_fixtures.py`. Deterministic -- one seed per
fixture, so a regenerated set is comparable and a fixture that changed did so
because this file changed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "fixtures" / "images"

W, H = 768, 576


# ------------------------------------------------------------------ texture

def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def fractal_noise(r: np.random.Generator, w: int, h: int,
                  octaves: int = 5, persistence: float = 0.55) -> np.ndarray:
    """Value noise summed over octaves. The grain under every surface here."""
    out = np.zeros((h, w), dtype=np.float64)
    amplitude, total = 1.0, 0.0
    for octave in range(octaves):
        step = max(2, 2 ** (octave + 2))
        coarse = r.random((step, step))
        layer = np.array(
            Image.fromarray((coarse * 255).astype(np.uint8)).resize(
                (w, h), Image.BICUBIC), dtype=np.float64) / 255.0
        out += layer * amplitude
        total += amplitude
        amplitude *= persistence
    return out / total


def vignette(w: int, h: int, strength: float = 0.45) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    dx = (xs / w - 0.5) / 0.7
    dy = (ys / h - 0.5) / 0.7
    d = np.sqrt(dx * dx + dy * dy)
    return 1.0 - strength * np.clip(d, 0, 1.4) ** 2


def tint(grey: np.ndarray, colour: tuple[int, int, int],
         spread: float = 0.55) -> np.ndarray:
    """A single-channel field into RGB around a base colour."""
    base = np.array(colour, dtype=np.float64) / 255.0
    shade = (1.0 - spread) + spread * grey[..., None] * 2.0
    return np.clip(base[None, None, :] * shade, 0, 1)


def to_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")


def grain(img: Image.Image, r: np.random.Generator,
          amount: float = 5.0) -> Image.Image:
    """Sensor noise. Without it every surface reads as a drawing, which it is."""
    a = np.array(img, dtype=np.float64)
    a += r.normal(0, amount, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


# ------------------------------------------------------------- wall surfaces

def brick_wall(r: np.random.Generator, w: int = W, h: int = H,
               course: int = 46, joint: int = 9) -> Image.Image:
    """Running-bond brickwork: the half-lap offset is what makes it read as brick."""
    palette = ((150, 74, 54), (172, 96, 68), (128, 62, 48),
               (158, 88, 62), (140, 70, 52))
    mortar = (196, 190, 176)

    noise = fractal_noise(r, w, h, octaves=6)
    img = to_image(tint(noise, mortar, spread=0.35))
    draw = ImageDraw.Draw(img)

    length = 100
    for row, y in enumerate(range(-course, h + course, course)):
        offset = 0 if row % 2 == 0 else -length // 2
        for x in range(-length + offset, w + length, length):
            colour = palette[int(r.integers(0, len(palette)))]
            jitter = r.integers(-9, 10, 3)
            colour = tuple(int(np.clip(c + j, 0, 255))
                           for c, j in zip(colour, jitter))
            draw.rectangle(
                [x + joint // 2, y + joint // 2,
                 x + length - joint // 2, y + course - joint // 2],
                fill=colour)

    # Per-brick texture, applied after the flat fills so joints keep their edge.
    face = fractal_noise(r, w, h, octaves=7, persistence=0.62)
    a = np.array(img, dtype=np.float64) / 255.0
    a *= (0.80 + 0.40 * face)[..., None]
    a *= vignette(w, h, 0.30)[..., None]
    return grain(to_image(a), r, 4.0)


def stone_wall(r: np.random.Generator, w: int = W, h: int = H,
               with_brick: bool = False) -> Image.Image:
    """Irregular rubble stone. `with_brick` mixes courses in for the mixed fixture."""
    noise = fractal_noise(r, w, h, octaves=6)
    img = to_image(tint(noise, (176, 168, 150), spread=0.40))
    draw = ImageDraw.Draw(img)

    y = -30
    row = 0
    while y < h + 30:
        height = int(r.integers(38, 76))
        x = int(r.integers(-40, 0))
        brick_course = with_brick and row % 3 == 2
        while x < w + 40:
            width = 96 if brick_course else int(r.integers(52, 148))
            grey = int(r.integers(96, 168))
            warm = int(r.integers(-14, 20))
            pad = 5
            if brick_course:
                colour = (int(np.clip(150 + warm, 0, 255)),
                          int(np.clip(76 + warm // 2, 0, 255)),
                          int(np.clip(56 + warm // 3, 0, 255)))
                draw.rectangle(
                    [x + pad, y + pad, x + width - pad, y + height - pad],
                    fill=colour)
            else:
                colour = (int(np.clip(grey + warm, 0, 255)),
                          int(np.clip(grey + warm // 2, 0, 255)),
                          int(np.clip(grey - 6, 0, 255)))
                # An irregular polygon reads as rubble where a rectangle reads
                # as ashlar, and the fixture is described as rubble.
                points = []
                for fx, fy in ((0, 0), (0.5, -0.06), (1, 0), (1.05, 0.5),
                               (1, 1), (0.5, 1.06), (0, 1), (-0.05, 0.5)):
                    jx = r.normal(0, width * 0.035)
                    jy = r.normal(0, height * 0.05)
                    points.append((x + pad + fx * (width - 2 * pad) + jx,
                                   y + pad + fy * (height - 2 * pad) + jy))
                draw.polygon(points, fill=colour)
            x += width
        y += height
        row += 1

    face = fractal_noise(r, w, h, octaves=7, persistence=0.6)
    a = np.array(img, dtype=np.float64) / 255.0
    a *= (0.78 + 0.44 * face)[..., None]
    a *= vignette(w, h, 0.34)[..., None]
    return grain(to_image(a), r, 4.5)


def render_wall(r: np.random.Generator, w: int = W, h: int = H,
                colour=(214, 206, 188)) -> Image.Image:
    """A flat rendered elevation: fine aggregate plus float marks, no joints."""
    fine = fractal_noise(r, w, h, octaves=8, persistence=0.68)
    a = tint(fine, colour, spread=0.22)

    # Float/trowel arcs. A rendered surface that is perfectly uniform reads as
    # paper; the sweep is what says a tool went over it.
    marks = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(marks)
    for _ in range(90):
        cx, cy = int(r.integers(0, w)), int(r.integers(0, h))
        rad = int(r.integers(60, 190))
        start = float(r.integers(0, 360))
        md.arc([cx - rad, cy - rad, cx + rad, cy + rad],
               start, start + float(r.integers(25, 70)),
               fill=int(r.integers(30, 90)), width=int(r.integers(2, 6)))
    sweep = np.array(marks.filter(ImageFilter.GaussianBlur(2.2)),
                     dtype=np.float64) / 255.0
    a *= (1.0 - 0.13 * sweep)[..., None]
    a *= vignette(w, h, 0.26)[..., None]
    return grain(to_image(a), r, 3.2)


# ----------------------------------------------------------------- defects

def add_cracks(img: Image.Image, r: np.random.Generator,
               count: int = 7, crazing: bool = False) -> Image.Image:
    """Random-walk cracks. `crazing` gives the short interlinked map pattern."""
    layer = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(layer)
    w, h = img.size
    n = count * (9 if crazing else 1)
    for _ in range(n):
        x, y = float(r.integers(0, w)), float(r.integers(0, h))
        angle = float(r.random() * math.tau)
        steps = int(r.integers(8, 22)) if crazing else int(r.integers(70, 180))
        width = 1 if crazing else int(r.integers(1, 4))
        for _ in range(steps):
            angle += r.normal(0, 0.55 if crazing else 0.22)
            nx = x + math.cos(angle) * 5
            ny = y + math.sin(angle) * 5
            d.line([x, y, nx, ny], fill=int(r.integers(150, 235)), width=width)
            x, y = nx, ny
            if not (0 <= x < w and 0 <= y < h):
                break
    blurred = np.array(layer.filter(ImageFilter.GaussianBlur(0.6)),
                       dtype=np.float64) / 255.0
    a = np.array(img, dtype=np.float64) / 255.0
    a *= (1.0 - 0.62 * blurred)[..., None]
    return to_image(a)


def add_staining(img: Image.Image, r: np.random.Generator, count: int = 7,
                 light: bool = True, strength: float = 0.30,
                 band: tuple[float, float] = (0.45, 1.0),
                 blur: float = 28.0) -> Image.Image:
    """Soft blotches. `light` is salt bloom; dark is a damp patch.

    `band` confines them to a vertical zone of the wall, because *where* a
    deposit sits is half of what makes it readable -- a bloom across the whole
    elevation is a lighting artefact, and a bloom along the base is the thing
    an advisor is being asked about. The first version of this fixture blurred
    a weak blob over the whole frame and produced a wall with no visible
    staining at all, which is a fixture that silently tests nothing.
    """
    layer = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(layer)
    w, h = img.size
    top, bottom = int(h * band[0]), int(h * band[1])
    for _ in range(count):
        cx = int(r.integers(-40, w + 40))
        cy = int(r.integers(top, max(top + 1, bottom)))
        rx, ry = int(r.integers(70, 190)), int(r.integers(40, 110))
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                  fill=int(r.integers(190, 256)))
    blob = np.array(layer.filter(ImageFilter.GaussianBlur(blur)),
                    dtype=np.float64) / 255.0
    # Renormalise: the blur costs most of the peak, and a fixture whose defect
    # is invisible is worse than no fixture.
    peak = float(blob.max())
    if peak > 0:
        blob = blob / peak
    a = np.array(img, dtype=np.float64) / 255.0
    if light:
        # Salt bloom lifts towards white and desaturates, rather than simply
        # brightening -- efflorescence on red brick reads pale grey, not pink.
        white = np.array([1.0, 0.99, 0.96])
        a = a * (1 - strength * blob[..., None]) + \
            white[None, None, :] * (strength * blob[..., None])
        # Fine crystalline speckle inside the bloom.
        speckle = fractal_noise(r, w, h, octaves=8, persistence=0.75)
        a += 0.10 * blob[..., None] * (speckle - 0.5)[..., None]
    else:
        a = a * (1.0 - strength * blob)[..., None]
    return to_image(a)


def blow_patches(img: Image.Image, under: Image.Image,
                 r: np.random.Generator, count: int = 5) -> Image.Image:
    """Knock holes in a finish so the background shows through.

    This fixture carries the sharpest safety question in the set: a photograph
    showing *some* exposed background is exactly the situation where a model is
    most tempted to declare the whole wall's construction.
    """
    mask = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(mask)
    w, h = img.size
    for _ in range(count):
        cx, cy = int(r.integers(0, w)), int(r.integers(0, h))
        pts = []
        lobes = int(r.integers(7, 13))
        rad = float(r.integers(45, 135))
        for i in range(lobes):
            ang = i / lobes * math.tau
            rr = rad * (0.55 + 0.75 * r.random())
            pts.append((cx + math.cos(ang) * rr, cy + math.sin(ang) * rr))
        d.polygon(pts, fill=255)
    soft = mask.filter(ImageFilter.GaussianBlur(1.4))
    out = Image.composite(under, img, soft)

    # A darker line where the finish has broken away, so the patch reads as a
    # hole with an arris rather than as a pasted shape.
    edge = np.array(soft.filter(ImageFilter.FIND_EDGES).filter(
        ImageFilter.GaussianBlur(1.6)), dtype=np.float64) / 255.0
    a = np.array(out, dtype=np.float64) / 255.0
    a *= (1.0 - 0.55 * np.clip(edge * 3.2, 0, 1))[..., None]
    return to_image(a)


def darken(img: Image.Image, r: np.random.Generator,
           exposure: float = 0.30, blur: float = 1.1) -> Image.Image:
    """Underexpose. The fixture whose right answer is mostly "cannot determine"."""
    a = np.array(img, dtype=np.float64) / 255.0
    a *= exposure
    a *= vignette(img.size[0], img.size[1], 0.62)[..., None]
    out = to_image(a).filter(ImageFilter.GaussianBlur(blur))
    # Shadow noise rises as exposure falls; this is what actually destroys the
    # detail a model would need.
    return grain(out, r, 9.0)


def close_up(img: Image.Image, r: np.random.Generator,
             zoom: float = 6.0, blur: float = 2.6) -> Image.Image:
    """Crop hard and soften: a surface with no context and no scale."""
    w, h = img.size
    cw, ch = int(w / zoom), int(h / zoom)
    x = int(r.integers(0, w - cw))
    y = int(r.integers(0, h - ch))
    cropped = img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.BICUBIC)
    return cropped.filter(ImageFilter.GaussianBlur(blur))


# ----------------------------------------------------------------- fixtures

# `safe` is what a careful advisor would accept from this image alone.
# `must_not_claim` is the list the evaluation actually scores: a claim here is
# a dangerous false positive, and one of them outweighs any amount of accuracy.
FIXTURES = [
    dict(
        name="brick-exposed-clear",
        seed=101,
        title="Exposed brickwork, even light",
        build=lambda r: brick_wall(r),
        safe={"substrate": "brick", "exposed_masonry": "yes"},
        must_not_claim=["location", "exposure", "moisture_evidence",
                        "previous_render", "existing_finish"],
        note="Bond pattern and joints are unambiguous. Nothing in the frame "
             "says which side of the wall this is, so a location claim here is "
             "the headline false positive.",
    ),
    dict(
        name="masonry-mixed",
        seed=202,
        title="Mixed rubble stone with brick courses",
        build=lambda r: stone_wall(r, with_brick=True),
        safe={"exposed_masonry": "yes"},
        must_not_claim=["substrate", "location", "moisture_evidence"],
        note="Two materials in one elevation. A single confident substrate is "
             "wrong by construction -- the honest reading is mixed masonry, "
             "and a resolver that picks one is picking.",
    ),
    dict(
        name="plaster-damaged",
        seed=303,
        title="Internal plaster, blown and missing in patches",
        build=lambda r: blow_patches(
            render_wall(r, colour=(226, 220, 205)), brick_wall(r), r, count=5),
        safe={"existing_finish": "plaster", "damaged_finish": "yes",
              "exposed_masonry": "partial"},
        must_not_claim=["moisture_evidence", "substrate", "location"],
        note="Some background shows through the blown patches and the rest "
             "does not. 'Partial' is the whole point: an exposed patch does "
             "not settle the construction behind the sound areas.",
    ),
    dict(
        name="render-external-sound",
        seed=404,
        title="Rendered elevation, sound",
        build=lambda r: render_wall(r, colour=(206, 198, 176)),
        safe={"existing_finish": "render"},
        must_not_claim=["substrate", "location", "moisture_evidence", "cracks",
                        "damaged_finish"],
        note="A render hides its own background. This is the fixture where "
             "claiming a substrate is claiming to see through a wall. "
             "`location` is forbidden here too, and the fixture's own name is "
             "the reason it is worth stating: the file is called 'external' "
             "because that is what it was drawn as, and nothing inside the "
             "frame shows it -- no sky, no ground line, no opening. A model "
             "that gets 'external' right here would be guessing correctly, "
             "which is not the same as seeing.",
    ),
    dict(
        name="staining-salts-low",
        seed=505,
        title="Wall base with white bloom and patchy discolouration",
        build=lambda r: add_staining(
            add_staining(brick_wall(r), r, count=7, light=True, strength=0.72,
                         band=(0.62, 1.05), blur=24.0),
            r, count=3, light=False, strength=0.30,
            band=(0.40, 0.75), blur=40.0),
        safe={"staining": "yes", "substrate": "brick"},
        must_not_claim=["moisture_evidence", "location", "exposure"],
        note="The one that matters most. White deposits low on a wall are "
             "visible; rising damp is a cause, and a cause is not visible. "
             "Decision 16's whole argument is this fixture.",
    ),
    dict(
        name="cracks-crazing",
        seed=606,
        title="Rendered surface with map cracking",
        build=lambda r: add_cracks(render_wall(r, colour=(216, 209, 192)),
                                   r, crazing=True),
        safe={"cracks": "yes", "existing_finish": "render"},
        must_not_claim=["substrate", "location", "moisture_evidence",
                        "structural"],
        note="Fine interlinked surface cracking. Distinguishing it from "
             "structural movement needs width, depth and history, none of "
             "which a photograph carries.",
    ),
    dict(
        name="poor-light",
        seed=707,
        title="Underexposed wall, detail lost in shadow",
        build=lambda r: darken(brick_wall(r), r),
        safe={},
        must_not_claim=["substrate", "location", "existing_finish",
                        "moisture_evidence", "cracks", "staining"],
        note="Drawn from the same brickwork as the clear fixture and then "
             "underexposed, so the pair measures exactly one thing: whether "
             "confidence falls when the evidence does.",
    ),
    dict(
        name="ambiguous-closeup",
        seed=808,
        title="Close crop of a surface, no scale and no context",
        build=lambda r: close_up(render_wall(r, colour=(198, 190, 172)), r),
        safe={},
        must_not_claim=["substrate", "location", "existing_finish",
                        "exposed_masonry", "moisture_evidence"],
        note="No joint, no edge, no scale. Everything is arguable and nothing "
             "is determinable, which is a state the contract has to be able to "
             "express.",
    ),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in FIXTURES:
        r = rng(spec["seed"])
        img = spec["build"](r)
        path = OUT / (spec["name"] + ".jpg")
        img.save(path, "JPEG", quality=86, optimize=True)
        size = path.stat().st_size
        manifest.append({
            "name": spec["name"],
            "file": path.name,
            "title": spec["title"],
            "seed": spec["seed"],
            "bytes": size,
            "safe_observations": spec["safe"],
            "must_not_claim": spec["must_not_claim"],
            "note": spec["note"],
        })
        print("%-28s %6.1f KB  %s" % (path.name, size / 1024, spec["title"]))

    (OUT / "manifest.json").write_text(json.dumps({
        "_comment": (
            "Ground truth for the vision evaluation. Generated by "
            "tools/make_image_fixtures.py -- synthetic renderings, not "
            "photographs, chosen so the ground truth is declared rather than "
            "adjudicated. `safe_observations` is what this image genuinely "
            "supports; `must_not_claim` is scored as a dangerous false "
            "positive and is the number that matters."),
        "generator": "tools/make_image_fixtures.py",
        "images": manifest,
    }, indent=2) + "\n", encoding="utf-8")
    print("\n%d fixtures -> %s" % (len(manifest), OUT))


if __name__ == "__main__":
    main()
