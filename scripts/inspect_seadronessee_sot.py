from pathlib import Path
import json

ROOT = Path(r"D:\datasets\SeaDronesSee")

json_path = ROOT / "sot" / "SeaDronesSee_train.json"
ann_path = ROOT / "sot" / "train_annotations" / "1.txt"

with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print("JSON path:", json_path)
print("Top-level type:", type(data).__name__)
print("Top-level len:", len(data))

selected_key = None
selected_value = None

if isinstance(data, dict):
    keys = list(data.keys())
    print("First 10 keys:", keys[:10])

    if "1" in data:
        selected_key = "1"
        selected_value = data["1"]
    elif 1 in data:
        selected_key = 1
        selected_value = data[1]
    else:
        selected_key = keys[0]
        selected_value = data[selected_key]

    print("Selected key:", selected_key)
    print("Selected value type:", type(selected_value).__name__)

    if isinstance(selected_value, list):
        print("Mapped frame count:", len(selected_value))
        print("First 10 mapped items:")
        for item in selected_value[:10]:
            print("  ", item)
    elif isinstance(selected_value, dict):
        print("Subkeys:", list(selected_value.keys())[:20])
        for k, v in selected_value.items():
            print(f"Example field: {k} -> {type(v).__name__}")
            if isinstance(v, list):
                print("  first 5:", v[:5])
            else:
                print("  value:", v)
            break

elif isinstance(data, list):
    print("First item type:", type(data[0]).__name__)
    print("First item:", data[0])

print("=" * 80)

lines = [x.strip() for x in ann_path.read_text(encoding="utf-8").splitlines() if x.strip()]
print("Annotation path:", ann_path)
print("Annotation line count:", len(lines))
print("First 10 annotation lines:")
for line in lines[:10]:
    print("  ", line)