# Split data into train, validation and test
import os
import math
import shutil
import glob
import cv2

# copy images from the mask folder that actually have masks
def copy_masks_having_masks(mask_dir, dest_dir):
    files = []
    for filename in glob.iglob(mask_dir + "**/*.png", recursive=True):
        image = cv2.imread(filename, 0)
        if cv2.countNonZero(image) == 0:
            pass
        else:
            files.append(filename)
    for i in files:
        shutil.copy(i, dest_dir)


# Copy into the images folder original images that have masks in the masks folder
# it returns an array of masks at the end
def copy_images_having_masks(images, masks, dest_dir):
    imgs = []
    msks = []
    for (dirpath, dirnames, filenames) in os.walk(images):
        imgs.extend(filenames)
    for (dirpath, dirnames, filenames) in os.walk(masks):
        msks.extend(filenames)
    print(len(imgs))
    print(len(msks))
    for p in imgs:
        if p in msks:
            # print("mask has image")
            shutil.copy(os.path.join(images, p), dest_dir)
    return msks


# split data into train, validation and test
def train_test_val_split(images_dir, mask_dir, train_dir, val_dir, test_dir, msks):
    n = len(msks)
    train_num = math.floor(0.8719 * n)
    # print (train_num)
    val_num = math.ceil(0.938967 * n)
    # print(val_num)
    train = msks[0:train_num]
    val = msks[train_num:val_num]
    test = msks[val_num : n + 1]

    # copy train images and masks into their respective folders
    for image_name in train:
        shutil.copy(os.path.join(images_dir, image_name), train_dir + "images")
        shutil.copy(os.path.join(mask_dir, image_name), train_dir + "masks")

    # copy val images and masks into their respective folders
    for image_name in val:
        shutil.copy(os.path.join(images_dir, image_name), val_dir + "images")
        shutil.copy(os.path.join(mask_dir, image_name), val_dir + "masks")

    # copy test images and masks into their respective folders
    for image_name in test:
        shutil.copy(os.path.join(images_dir, image_name), test_dir + "images")
        shutil.copy(os.path.join(mask_dir, image_name), test_dir + "masks")

    print("Done!")
