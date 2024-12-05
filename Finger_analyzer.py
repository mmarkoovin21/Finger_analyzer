import os
import cv2

def sort_images(source_dir, train_dir, test_dir):
    create_directories_if_not_exist(train_dir, test_dir)

    image_files = []
    for file in os.listdir(source_dir):
        if file.endswith('.tif'):
            image_files.append(file)

    for index, file in enumerate(image_files, start=1):
        image_source = os.path.join(source_dir, file)
        processed_image = process_image(image_source)
        if index % 8 == 0:
            cv2.imwrite(os.path.join(test_dir, file), processed_image)
        else:
            cv2.imwrite(os.path.join(train_dir, file), processed_image)

def create_directories_if_not_exist(*directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def process_image(image_path):
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return threshold_image

def main():
    source_directory = './SourceFolder'
    training_directory = './TrainingSet'
    testing_directory = './TestSet'

    create_directories_if_not_exist(source_directory, training_directory, testing_directory)

    if not os.listdir(source_directory):
        print("Source directory is empty! Fill it with fingerprint database.")
        return

    if os.listdir(training_directory) or os.listdir(testing_directory):
        print("Images already sorted and processed.")
    else:
        sort_images(source_directory, training_directory, testing_directory)
        print("Images have been sorted and processed.")

if __name__ == "__main__":
    main()
