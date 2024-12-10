import os
import cv2
import shutil
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def process_and_prepare_data(source_dir, processed_dir):
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    images, labels = load_and_process_images(source_dir, processed_dir)
    normalized_images = preprocessing.normalize(images, norm='l2')
    return labels, normalized_images

def load_and_process_images(source_dir, processed_dir):
    image_data = []
    image_labels = []

    for file in os.listdir(source_dir):
        if file.endswith('.tif'):
            file_path = os.path.join(source_dir, file)
            processed_image = preprocess_image(file_path)
            flattened_image = np.array(processed_image).flatten()
            image_data.append(flattened_image)

            label = extract_label(file)
            image_labels.append(label)

            save_image(processed_image, file, processed_dir)

    return image_data, image_labels

def preprocess_image(image_path):
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    equalized_image = cv2.equalizeHist(gray_image)
    gabor_filtered = apply_gabor_filter(equalized_image)
    thresholded_image = cv2.threshold(gabor_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    normalized_image = cv2.normalize(thresholded_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized_image

def apply_gabor_filter(image):
    kernel_size = 31
    sigma = 4.0
    theta = 0
    lambd = 10.0
    gamma = 0.5
    psi = 0

    gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)

    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    return filtered_image

def save_image(image, filename, dest_dir):
    cv2.imwrite(os.path.join(dest_dir, filename), image)

def extract_label(filename):
    return filename.split('_')[0]

def compute_prediction_probabilities(test_data, model, threshold):
    decision_scores = np.array(model.decision_function(test_data))
    probabilities = np.exp(decision_scores) / np.sum(np.exp(decision_scores), axis=1, keepdims=True)
    thresholded_predictions = (np.max(probabilities, axis=1) > threshold).astype(int)
    return thresholded_predictions

def create_directories_if_not_exist(*directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def main():
    fingerprints_source = './Fingerprints_DB'
    processed_fingerprints_dir = './Processed_Fingerprints'
    create_directories_if_not_exist(fingerprints_source, processed_fingerprints_dir)

    if not os.listdir(fingerprints_source):
        print("Fingerprint source directory is empty! Please add fingerprint data.")
        return

    labels, normalized_data = process_and_prepare_data(fingerprints_source, processed_fingerprints_dir)

    train_data, test_data, train_labels, test_labels = train_test_split(
        normalized_data, labels, test_size=0.2, random_state=42
    )

    print("Fingerprints have been processed and split into training and testing sets. Starting model training.")

    model = OneVsRestClassifier(svm.SVC(kernel='rbf', C=10.0, gamma=1.0, decision_function_shape='ovr'))
    model.fit(train_data, train_labels)

    predicted_labels = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"Training completed. Model accuracy: {accuracy * 100:.2f}%")

    threshold = 0.1
    probabilities = compute_prediction_probabilities(test_data, model, threshold)

    correct_predictions = sum(1 for i in range(len(test_labels)) if test_labels[i] == predicted_labels[i] and probabilities[i] == 1)
    frr = 1 - (correct_predictions / len(test_labels))
    print(f"False Rejection Rate (FRR): {frr:.2%}")

    fake_fingerprints_source = './Imposter_Fingerprints_DB'
    processed_fake_fingerprints_dir = './Processed_Imposter_Fingerprints'
    create_directories_if_not_exist(fake_fingerprints_source, processed_fake_fingerprints_dir)

    if not os.listdir(fake_fingerprints_source):
        print("Skipping FAR calculation as fake fingerprint database is empty.")
        return

    fake_labels, fake_normalized_data = process_and_prepare_data(fake_fingerprints_source, processed_fake_fingerprints_dir)

    fake_probabilities = compute_prediction_probabilities(fake_normalized_data, model, threshold)
    false_acceptances = sum(1 for i in range(len(fake_labels)) if fake_probabilities[i] == 1)
    far = false_acceptances / len(test_labels)
    print(f"False Acceptance Rate (FAR): {far:.2%}")


if __name__ == "__main__":
    main()
