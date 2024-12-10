# This file is part of Finger_analyzer.
# Copyright (C) 2024 Magdalena Markovinović.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import cv2
import shutil
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FingerprintProcessor:
    def __init__(self, source_dir, processed_dir):
        self.source_dir = source_dir
        self.processed_dir = processed_dir
        self.image_data = []
        self.labels = []

    def process_and_prepare_data(self):
        self._clear_directory(self.processed_dir)
        self._load_and_process_images()
        normalized_images = preprocessing.normalize(self.image_data, norm='l2')
        return self.labels, normalized_images

    def _load_and_process_images(self):
        for file in os.listdir(self.source_dir):
            if file.endswith('.tif'):
                file_path = os.path.join(self.source_dir, file)
                processed_image = self._preprocess_image(file_path)
                flattened_image = np.array(processed_image).flatten()
                self.image_data.append(flattened_image)

                label = self._extract_label(file)
                self.labels.append(label)

                self._save_image(processed_image, file)

    def _preprocess_image(self, image_path):
        target_width = 326
        target_height = 357
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(gray_image, (target_width, target_height))
        gabor_filtered = self._apply_gabor_filter(resized_image)
        equalized_image = cv2.equalizeHist(gabor_filtered)
        
        thresholded_image = cv2.threshold(equalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        normalized_image = cv2.normalize(thresholded_image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized_image

    @staticmethod
    def _apply_gabor_filter(image):
        kernel_size = 31
        sigma = 5.0
        theta = 0
        lambd = 10.0
        gamma = 0.2
        psi = 0

        gabor_kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
        )
        return cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)

    def _save_image(self, image, filename):
        cv2.imwrite(os.path.join(self.processed_dir, filename), image)

    @staticmethod
    def _extract_label(filename):
        return filename.split('_')[0]

    @staticmethod
    def _clear_directory(directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)


class FingerprintModel:
    def __init__(self, threshold=0.285):
        self.threshold = threshold
        self.model = OneVsRestClassifier(svm.SVC(kernel='rbf', C=35, gamma=0.5, decision_function_shape='ovr'))

    def train(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def evaluate(self, test_data, test_labels):
        predicted_labels = self.model.predict(test_data)
        accuracy = accuracy_score(test_labels, predicted_labels)
        return predicted_labels, accuracy

    def compute_prediction_probabilities(self, data):
        decision_scores = np.array(self.model.decision_function(data))
        probabilities = np.exp(decision_scores) / np.sum(np.exp(decision_scores), axis=1, keepdims=True)
        thresholded_predictions = (np.max(probabilities, axis=1) > self.threshold).astype(int)
        return thresholded_predictions


def main():
    fingerprints_source = './Fingerprints_DB'
    processed_fingerprints_dir = './Processed_Fingerprints'
    fake_fingerprints_source = './Imposter_Fingerprints_DB'
    processed_fake_fingerprints_dir = './Processed_Imposter_Fingerprints'

    for directory in [fingerprints_source, processed_fingerprints_dir, fake_fingerprints_source, processed_fake_fingerprints_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    if not os.listdir(fingerprints_source):
        print(f"The directory Fingerprints_DB is empty! Please add fingerprint data.")
        return

    if not os.listdir(fake_fingerprints_source):
        print(f"The directory Imposter_Fingerprints_DB is empty! Please add imposter fingerprint data.")
        return

    fingerprint_processor = FingerprintProcessor(fingerprints_source, processed_fingerprints_dir)
    labels, normalized_data = fingerprint_processor.process_and_prepare_data()

    train_data, test_data, train_labels, test_labels = train_test_split(
        normalized_data, labels, test_size=0.2, random_state=42
    )

    print("Fingerprints have been processed and split into training and testing sets. Starting model training.")

    model = FingerprintModel()
    model.train(train_data, train_labels)

    predicted_labels, accuracy = model.evaluate(test_data, test_labels)
    print(f"Training completed. Model accuracy: {accuracy * 100:.2f}%")

    probabilities = model.compute_prediction_probabilities(test_data)
    correct_predictions = sum(
        1
        for i in range(len(test_labels))
        if test_labels[i] == predicted_labels[i] and probabilities[i] == 1
    )
    frr = 1 - (correct_predictions / len(test_labels))
    print(f"False Rejection Rate (FRR): {frr:.2%}")

    fake_processor = FingerprintProcessor(fake_fingerprints_source, processed_fake_fingerprints_dir)
    fake_labels, fake_normalized_data = fake_processor.process_and_prepare_data()

    fake_probabilities = model.compute_prediction_probabilities(fake_normalized_data)
    false_acceptances = sum(1 for i in range(len(fake_labels)) if fake_probabilities[i] == 1)
    far = false_acceptances / len(fake_labels)
    print(f"False Acceptance Rate (FAR): {far:.2%}")


if __name__ == "__main__":
    main()
