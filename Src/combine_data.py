import os
import shutil
from sklearn.model_selection import train_test_split


final_base = "final_dataset"

# Categories 
categories = ['cats', 'dogs', 'fruits', 'tiger_lion', 'birds']

# Create train, val, and test folders
for split in ['train', 'val', 'test']:
    for cat in categories:
        os.makedirs(os.path.join(final_base, split, cat), exist_ok=True)


def copy_images(src_dir, dest_dir):
    if not os.path.exists(src_dir):
        print(f"Skipping (not found): {src_dir}")
        return
    count = 0
    for img in os.listdir(src_dir):
        src_path = os.path.join(src_dir, img)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_dir)
            count += 1
    print(f"Copied {count} images from {src_dir} → {dest_dir}")

# Cats & Dogs dataset
copy_images("data/training_set/training_set/cats", "final_dataset/train/cats")
copy_images("data/training_set/training_set/dogs", "final_dataset/train/dogs")

# Birds  (all subfolders)
bird_base = "data/Bird Speciees Dataset"
if os.path.exists(bird_base):
    for bclass in os.listdir(bird_base):
        src_folder = os.path.join(bird_base, bclass)
        if os.path.isdir(src_folder):
            copy_images(src_folder, "final_dataset/train/birds")

# Tiger & Lion dataset
tiger_base = "data/tiger&lion"
if os.path.exists(tiger_base):
    for species in os.listdir(tiger_base):
        src_folder = os.path.join(tiger_base, species)
        if os.path.isdir(src_folder):
            copy_images(src_folder, "final_dataset/train/tiger_lion")

# Fruits dataset 
fruits_base = "data/images"
if os.path.exists(fruits_base):
    for folder in os.listdir(fruits_base):
        src_folder = os.path.join(fruits_base, folder)
        if os.path.isdir(src_folder):
            copy_images(src_folder, "final_dataset/train/fruits")

# Split data 
def create_splits(category):
    train_dir = os.path.join(final_base, "train", category)
    val_dir = os.path.join(final_base, "val", category)
    test_dir = os.path.join(final_base, "test", category)

    all_imgs = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    if len(all_imgs) < 10:
        print(f"Skipping {category}, not enough images.")
        return

    # Split data set
    train_imgs, temp_imgs = train_test_split(all_imgs, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    for img in val_imgs:
        shutil.move(os.path.join(train_dir, img), os.path.join(val_dir, img))
    for img in test_imgs:
        shutil.move(os.path.join(train_dir, img), os.path.join(test_dir, img))

    print(f"Split done for {category}: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")


for cat in categories:
    create_splits(cat)

print("\nFull dataset (train/val/test) ready in 'final_dataset/'")
