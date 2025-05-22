import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Step 1: Generate synthetic dataset
def generate_shape(shape, size=64):
    image = np.zeros((size, size), dtype=np.uint8)
    if shape == "circle":
        cv2.circle(image, (size//2, size//2), size//4, 255, -1)
    elif shape == "square":
        cv2.rectangle(image, (size//4, size//4), (3*size//4, 3*size//4), 255, -1)
    elif shape == "triangle":
        points = np.array([[size//2, size//4], [size//4, 3*size//4], [3*size//4, 3*size//4]])
        cv2.drawContours(image, [points], 0, 255, -1)
    return image

def create_dataset(num_samples=1000):
    shapes = ["circle", "square", "triangle"]
    images, labels = [], []
    for _ in range(num_samples):
        shape = np.random.choice(shapes)
        image = generate_shape(shape)
        images.append(image)
        labels.append(shapes.index(shape))
    return np.array(images), np.array(labels)

# Generate dataset
images, labels = create_dataset()
images = images / 255.0  # Normalize pixel values
images = images.reshape(-1, 64, 64, 1)  # Reshape for CNN input
labels = to_categorical(labels, num_classes=3)  # One-hot encode labels

# Step 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 3: Build the ML model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 5: Test the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 6: Predict shapes from new images
def predict_shape(image):
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = image.reshape(1, 64, 64, 1)
    prediction = model.predict(image)
    classes = ["circle", "square", "triangle"]
    return classes[np.argmax(prediction)]

# Example usage with new image
#generated image
sample_image = generate_shape("triangle")

#user input images
# Load the image from file
# image_path = "shape.png"  # Change this to your image path
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale

# # Resize image to 64x64 (the size your model expects)
# image = cv2.resize(image, (64, 64))

# # Normalize pixel values
# image = image / 255.0

# # Reshape the image to match the input shape for the CNN
# image = image.reshape(1, 64, 64, 1)  # Add batch dimension

#prediction
recognized_shape = predict_shape(sample_image)
cv2.print(f"Recognized Shape: {recognized_shape}")
