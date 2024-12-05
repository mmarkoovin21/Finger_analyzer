import os
import shutil
import cv2
import numpy as np

def sort_images(source_dir, train_dir, test_dir):
    image_files = []
    for file in os.listdir(source_dir):
        if file.endswith('.tif'):
            image_files.append(file)

    for index, file in enumerate(image_files, start=1):
        if index % 8 == 0:
            shutil.copy2(os.path.join(source_dir, file), test_dir)
        else:
            shutil.copy2(os.path.join(source_dir, file), train_dir)

def main():
    source_directory = './SourceFolder'
    training_directory = './TrainingSet'
    testing_directory = './TestSet'

    if os.listdir(training_directory) or os.listdir(testing_directory):
        print("Images already sorted.")
    else:
        sort_images(source_directory, training_directory, testing_directory)

if __name__ == "__main__":
    main()
