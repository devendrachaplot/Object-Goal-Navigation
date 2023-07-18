scenes = {}
scenes["train"] = [
    "Allensville",
    "Beechwood",
    "Benevolence",
    "Coffeen",
    "Cosmos",
    "Forkland",
    "Hanson",
    "Hiteman",
    "Klickitat",
    "Lakeville",
    "Leonardo",
    "Lindenwood",
    "Marstons",
    "Merom",
    "Mifflinburg",
    "Newfields",
    "Onaga",
    "Pinesdale",
    "Pomaria",
    "Ranchester",
    "Shelbyville",
    "Stockman",
    "Tolstoy",
    "Wainscott",
    "Woodbine",
]

scenes["val"] = [
    "Collierville",
    "Corozal",
    "Darden",
    "Markleeville",
    "Wiconisco",
]

coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14,
}

coco_categories_replica = {
    "chair": 0,
    "sofa": 1,
    "plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "table": 6,
    "oven": 7,
    "sink": 8,
    "fridge": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14,
    "person": 15,
}

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
    0: 15,  # person
}

color_palette = [
    1.0,
    1.0,
    1.0,
    0.6,
    0.6,
    0.6,
    0.95,
    0.95,
    0.95,
    0.96,
    0.36,
    0.26,
    0.12156862745098039,
    0.47058823529411764,
    0.7058823529411765,
    0.9400000000000001,
    0.7818,
    0.66,
    0.9400000000000001,
    0.8868,
    0.66,
    0.8882000000000001,
    0.9400000000000001,
    0.66,
    0.7832000000000001,
    0.9400000000000001,
    0.66,
    0.6782000000000001,
    0.9400000000000001,
    0.66,
    0.66,
    0.9400000000000001,
    0.7468000000000001,
    0.66,
    0.9400000000000001,
    0.8518000000000001,
    0.66,
    0.9232,
    0.9400000000000001,
    0.66,
    0.8182,
    0.9400000000000001,
    0.66,
    0.7132,
    0.9400000000000001,
    0.7117999999999999,
    0.66,
    0.9400000000000001,
    0.8168,
    0.66,
    0.9400000000000001,
    0.9218,
    0.66,
    0.9400000000000001,
    0.9400000000000001,
    0.66,
    0.8531999999999998,
    0.9400000000000001,
    0.66,
    0.748199999999999,
]
