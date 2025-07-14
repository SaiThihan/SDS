import os

# Define paths
base_dir = "C:/Users/Khin Maung Thant/Desktop/Shan State Day Projects/Snake_Webcam_Detection/data"

splits = ["train", "val", "test"]

# Check images and labels
for split in splits:
    img_dir = os.path.join(base_dir, "images", split)
    label_dir = os.path.join(base_dir, "labels", split)

    print(f"Checking {split} split:")
    if not os.path.exists(img_dir):
        print(f"Image directory does not exist: {img_dir}")
    elif not os.path.exists(label_dir):
        print(f"Label directory does not exist: {label_dir}")
    else:
        img_files = set(os.listdir(img_dir))
        label_files = set(os.listdir(label_dir))

        # Remove file extensions for comparison
        img_names = {os.path.splitext(f)[0] for f in img_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        label_names = {os.path.splitext(f)[0] for f in label_files if f.lower().endswith('.txt')}

        # Find missing files
        missing_labels = img_names - label_names
        missing_images = label_names - img_names

        if missing_labels:
            print(f"Missing labels for images: {missing_labels}")
        if missing_images:
            print(f"Missing images for labels: {missing_images}")
        if not missing_labels and not missing_images:
            print(f"All {len(img_names)} images have corresponding labels.")