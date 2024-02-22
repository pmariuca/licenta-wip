import os
import cv2
import random


def create_directories(*directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def split_dataset(dataset_dir, train_dir, test_dir, train_percentage=0.8):
    create_directories(train_dir, test_dir)

    people_folders = os.listdir(dataset_dir)

    for people in people_folders:
        images = os.listdir(os.path.join(dataset_dir, people))
        random.shuffle(images)

        num_train_images = int(train_percentage * len(images))

        for i, image in enumerate(images):
            path = os.path.join(dataset_dir, people, image)
            dir = train_dir if i < num_train_images else test_dir
            dest_path = os.path.join(dir, people, image)
            create_directories(os.path.join(dir, people))
            cv2.imwrite(dest_path, cv2.imread(path))


if __name__ == "__main__":
    dataset_dir = "dataset"
    train_dir = "training"
    test_dir = "test"
    split_dataset(dataset_dir, train_dir, test_dir)
