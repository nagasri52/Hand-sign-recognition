# import os
# import shutil
# import random

# def create_validation_set(train_dir, val_dir, validation_split=0.2):
#     """
#     Creates a validation set from a training dataset organized in ImageFolder format.

#     Args:
#         train_dir (str): Path to the training dataset directory.
#         val_dir (str): Path to the directory where the validation set will be created.
#         validation_split (float): The proportion of images to move to the validation set.
#     """

#     if not os.path.exists(val_dir):
#         os.makedirs(val_dir)

#     class_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]

#     for class_folder in class_folders:
#         train_class_dir = os.path.join(train_dir, class_folder)
#         val_class_dir = os.path.join(val_dir, class_folder)

#         if not os.path.exists(val_class_dir):
#             os.makedirs(val_class_dir)

#         images = [f for f in os.listdir(train_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
#         random.shuffle(images)

#         num_val_images = int(len(images) * validation_split)
#         val_images = images[:num_val_images]

#         for image in val_images:
#             src_path = os.path.join(train_class_dir, image)
#             dest_path = os.path.join(val_class_dir, image)
#             shutil.move(src_path, dest_path)

#         print(f"Moved {num_val_images} images from {class_folder} to validation set.")

# if __name__ == "__main__":
#     train_directory = r'C:\Users\nagas\OneDrive\Desktop\New folder\dataset\Train'  # Replace with your training dataset path
#     validation_directory = r'C:\Users\nagas\OneDrive\Desktop\New folder\dataset\validation' # Replace with your validation dataset path
#     validation_percentage = 0.2  # Adjust the split percentage as needed

#     create_validation_set(train_directory, validation_directory, validation_percentage)

#     print("Validation set creation complete.")



import os
import shutil

# Paths
source_train = "dataset/train"
source_val = "dataset/validation"
dest_train = "dataset/images/train"
dest_val = "dataset/images/val"

# Create new directories
os.makedirs(dest_train, exist_ok=True)
os.makedirs(dest_val, exist_ok=True)

# Move training images
for class_folder in os.listdir(source_train):
    class_path = os.path.join(source_train, class_folder)
    if os.path.isdir(class_path):
        for img in os.listdir(class_path):
            shutil.move(os.path.join(class_path, img), dest_train)

# Move validation images
for class_folder in os.listdir(source_val):
    class_path = os.path.join(source_val, class_folder)
    if os.path.isdir(class_path):
        for img in os.listdir(class_path):
            shutil.move(os.path.join(class_path, img), dest_val)

print("Dataset structure fixed ✅")
