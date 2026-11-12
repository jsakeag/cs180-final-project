"""What we look for, and how big things usually are.

PROMPTS: text prompts given to the open-vocabulary detector.
LABEL_TO_CLASS: detector label -> engine class (several prompts can share one class).
SIZE_PRIORS: typical [width, height, depth] in meters. Width = the long side seen from
the front. Measured values are clamped to a range around these so one bad box or bad
depth reading can't produce a 6 m chair.
FLOOR_CLASSES: things that stand on the floor; only these are placed via the floor
homography. Others (tv, picture) are kept in detections but not placed yet.
"""

PROMPTS = [
    "bed", "chair", "sofa", "armchair", "dining table", "coffee table", "desk",
    "bookshelf", "bookcase", "floor lamp", "potted plant", "cabinet", "dresser", "nightstand",
    "wardrobe", "tv stand", "stool", "bench", "basket", "refrigerator", "toilet",
    "bathtub", "tv", "mirror", "picture frame", "window", "door",
]

LABEL_TO_CLASS = {
    "bed": "bed", "chair": "chair", "sofa": "sofa", "armchair": "armchair",
    "dining table": "table", "coffee table": "coffee_table", "desk": "desk",
    "bookshelf": "bookshelf", "bookcase": "bookshelf", "floor lamp": "lamp", "potted plant": "plant",
    "cabinet": "cabinet", "dresser": "dresser", "nightstand": "nightstand",
    "wardrobe": "wardrobe", "tv stand": "tv_stand", "stool": "stool", "bench": "bench",
    "basket": "basket", "refrigerator": "fridge", "toilet": "toilet", "bathtub": "bathtub",
    "tv": "tv", "mirror": "mirror", "picture frame": "picture", "window": "window", "door": "door",
}

SIZE_PRIORS = {
    "bed": [1.5, 0.6, 2.0],
    "chair": [0.5, 0.9, 0.5],
    "sofa": [2.0, 0.85, 0.9],
    "armchair": [0.85, 0.9, 0.85],
    "table": [1.4, 0.75, 0.9],
    "coffee_table": [1.0, 0.45, 0.6],
    "desk": [1.2, 0.75, 0.6],
    "bookshelf": [0.9, 1.8, 0.35],
    "lamp": [0.35, 1.5, 0.35],
    "plant": [0.4, 0.9, 0.4],
    "cabinet": [0.9, 1.0, 0.45],
    "dresser": [1.1, 0.9, 0.5],
    "nightstand": [0.5, 0.55, 0.4],
    "wardrobe": [1.0, 1.9, 0.6],
    "tv_stand": [1.6, 0.5, 0.45],
    "stool": [0.4, 0.65, 0.4],
    "bench": [1.2, 0.45, 0.4],
    "basket": [0.45, 0.4, 0.35],
    "fridge": [0.75, 1.8, 0.7],
    "toilet": [0.4, 0.75, 0.7],
    "bathtub": [1.6, 0.55, 0.75],
}

FLOOR_CLASSES = set(SIZE_PRIORS)

# measured / prior must land in this range or it's clamped
CLAMP = (0.6, 1.6)
