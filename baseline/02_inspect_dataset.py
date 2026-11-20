from collections import Counter

from wilds import get_dataset

from settings import DATA

dataset = get_dataset(dataset="camelyon17", root_dir=str(DATA), download=False)
center = dataset.metadata_fields.index("hospital")
for split in ["train", "id_val", "val", "test"]:
    subset = dataset.get_subset(split)
    hospitals = Counter(subset.metadata_array[:, center].tolist())
    print(split, "patches:", len(subset), "patches per hospital:", dict(hospitals))
    if split != "test":
        print("  class counts:", dict(Counter(subset.y_array.tolist())))
