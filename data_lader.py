import os # door mapstructuur te lopen
import cv2 # afbeeldingen lezen en resizen
import numpy as np # voor matrixbewerkingen

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

data_dir = r'c:\Users\Rachid\Documents\GTSRB_Dataset\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images'

# hier begint de functie om data in te laden
def load_data(data_dir, img_size=(32, 32)):
    images = []
    labels = []

    # Doorloop alle submappen (00000 t/m 00042)
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if file.endswith('.ppm'):
                img_path = os.path.join(label_path, file)
                image = cv2.imread(img_path) # lees de afbeelding als een matrix van pixels
                image = cv2.resize(image, img_size) # geef afbeelding deze size
                images.append(image) # afbeelding aan images lijst toevoegen
                labels.append(int(label))  # label omgezet naar getal en toegevoegd aan label array

    X = np.array(images) #np arrays
    y = np.array(labels)

    # Normaliseer en one-hot encode
    X = X / 255.0
    y = to_categorical(y, num_classes=43)

    # Train/test split
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Laad de data
X_train, X_test, y_train, y_test = load_data(data_dir)

# Print de grootte van de geladen data om te verifiëren
print(f"Aantal trainingsvoorbeelden: {len(X_train)}")
print(f"Aantal testvoorbeelden: {len(X_test)}")