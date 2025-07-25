import json
import os
TRAIN_PATH = os.path.expanduser("~/Desktop/textdetox/data/splits/zh_train_split.json")
DEV_PATH   = os.path.expanduser("~/Desktop/textdetox/data/splits/zh_dev_split.json")
OUTPUT     = "zh_detox_corpus.txt"

def load_entries(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        else:
            entries = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
            return entries

train_entries = load_entries(TRAIN_PATH)
dev_entries   = load_entries(DEV_PATH)

with open(OUTPUT, "w", encoding="utf-8") as out:
    for e in train_entries + dev_entries:
        src = e.get("source", "").strip()
        tgt = e.get("target", "").strip()
        if not src or not tgt:
            continue
        out.write(src + " " + tgt + "\n")

print(f"Wrote {len(train_entries) + len(dev_entries)} lines to {OUTPUT}")
