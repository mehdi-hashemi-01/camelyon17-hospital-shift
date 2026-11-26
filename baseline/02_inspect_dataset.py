from collections import Counter

from settings import load_camelyon17

dataset = load_camelyon17()
center = dataset.metadata_fields.index("hospital")
for split in ["train", "id_val", "val", "test"]:
    subset = dataset.get_subset(split)
    hospitals = Counter(subset.metadata_array[:, center].tolist())
    print(split, "patches:", len(subset), "patches per hospital:", dict(hospitals))
    if split != "test":
        print("  class counts:", dict(Counter(subset.y_array.tolist())))
