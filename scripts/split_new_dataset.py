import os
import shutil
import random

# Paths
base_dir = r"C:\Users\ADMIN\Desktop\bone-fracture-hybrid\new Dataset"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

splits = ["train", "val", "test"]
fracture_dir = {split: os.path.join(base_dir, split, "fracture") for split in splits}
no_fracture_dir = {split: os.path.join(base_dir, split, "no_fracture") for split in splits}

# Helper to get class from label file
def get_class(label_path):
    try:
        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                print(f"[WARN] Empty label file: {label_path}")
                return None
            if first_line.startswith('1 '):
                return 'fracture'
            elif first_line.startswith('0 '):
                return 'no_fracture'
            else:
                # fallback: check first token
                token = first_line.split()[0]
                if token == '1':
                    return 'fracture'
                elif token == '0':
                    return 'no_fracture'
                else:
                    print(f"[WARN] Unknown label in {label_path}: '{first_line}'")
                    return None
    except Exception as e:
        print(f"[ERROR] Failed to read label {label_path}: {e}")
        return None

# Gather all image-label pairs
pairs = []
skipped = 0
for fname in os.listdir(images_dir):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        label_fname = os.path.splitext(fname)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_fname)
        if os.path.exists(label_path):
            cls = get_class(label_path)
            if cls is not None:
                pairs.append((fname, label_fname, cls))
            else:
                skipped += 1
        else:
            print(f"[WARN] No label for image: {fname}")
            skipped += 1
print(f"[INFO] Total image-label pairs: {len(pairs)} (skipped: {skipped})")


# Shuffle and split
random.seed(42)
random.shuffle(pairs)
fracture = [p for p in pairs if p[2] == 'fracture']
no_fracture = [p for p in pairs if p[2] == 'no_fracture']
print(f"[INFO] Fracture: {len(fracture)}, No Fracture: {len(no_fracture)}")

def split_list(lst):
    n = len(lst)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train = lst[:n_train]
    val = lst[n_train:n_train+n_val]
    test = lst[n_train+n_val:]
    return train, val, test

fracture_split = split_list(fracture)
no_fracture_split = split_list(no_fracture)

# Copy files
def copy_files(split_idx, split_name, class_name, split_list):
    count = 0
    for img_fname, label_fname, _ in split_list:
        src_img = os.path.join(images_dir, img_fname)
        src_label = os.path.join(labels_dir, label_fname)
        dst_dir = fracture_dir[split_name] if class_name == 'fracture' else no_fracture_dir[split_name]
        try:
            shutil.copy2(src_img, os.path.join(dst_dir, img_fname))
            shutil.copy2(src_label, os.path.join(dst_dir, label_fname))
            count += 1
        except Exception as e:
            print(f"[ERROR] Failed to copy {img_fname} or {label_fname} to {dst_dir}: {e}")
    print(f"[INFO] Copied {count} {class_name} files to {split_name}/{class_name}")

for split_idx, split_name in enumerate(splits):
    copy_files(split_idx, split_name, 'fracture', fracture_split[split_idx])
    copy_files(split_idx, split_name, 'no_fracture', no_fracture_split[split_idx])

print("Dataset split and copy complete.")
