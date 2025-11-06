import os
import shutil
from sklearn.model_selection import train_test_split

base = "final_dataset"
categories = ['cats', 'dogs', 'fruits', 'tiger_lion', 'birds']

for cat in categories:
    
    src_dir = os.path.join(base, "val", cat)
    if not os.path.exists(src_dir):
        src_dir = os.path.join(base, "train", cat)

    test_dir = os.path.join(base, "test", cat)
    os.makedirs(test_dir, exist_ok=True)

    images = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    if len(images) == 0:
        print(f"⚠️ No images found for {cat}, skipping...")
        continue

    _, test_imgs = train_test_split(images, test_size=0.15, random_state=42)

    for img in test_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(test_dir, img))

    print(f"Created test split for {cat}: {len(test_imgs)} images")

print("\n All test splits created successfully!")
