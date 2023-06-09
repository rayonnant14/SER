DATASETS = {
    "SAVEE": {
        "num_classes": 7,
        "target_names": [
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        ],
        "seed": 44,
        "dilations": 8,
    },
    "IEMOCAP": {
        "num_classes": 4,
        "target_names": [
            "neutral",
            # "frustration",
            "sadness",
            # "surprise",
            "anger",
            "happiness",
            # "excitement",
            # "fear",
            # "disgust",
            # "other",
        ],
        "seed": 16,
        "dilations": 10,
    },
    "DUSHA": {
        "num_classes": 5,
        "target_names": [
            "positive",
            "sadness",
            "anger",
            "neutral",
            "other",
        ],
        "seed": 0,
        "dilations": 10,
    },
    "EMODB": {
        "num_classes": 7,
        "target_names": [
            "anger",
            "boredom",
            "disgust",
            "anxiety",
            "happiness",
            "sadness",
            "neutral",
        ],
        "seed": 46,
        "dilations": 8,
    },
    "EMOVO": {
        "num_classes": 7,
        "target_names": [
            "disgust",
            "joy",
            "fear",
            "anger",
            "surprise",
            "sad",
            "neutral",
        ],
        "seed": 1,
        "dilations": 8,
    },
    "RAVDESS": {
        "num_classes": 8,
        "target_names": [
            "neutral",
            "calm",
            "happy",
            "sad",
            "angry",
            "fearful",
            "disgust",
            "surprised",
        ],
        "seed": 46,
        "dilations": 8
    },
    "CROSS": {
        "num_classes": 6,
        "target_names": [
            "angry",
            "disgust",
            "fearful",
            "happy",
            "sad",
            "neutral",
        ],
        "seed": 0,
        "dilations": 8
    },
}
