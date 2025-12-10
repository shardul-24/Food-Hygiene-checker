# import os
# import cv2
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# import warnings
# warnings.filterwarnings('ignore')

# # Set path to your dataset folder with subfolders: good/, moderate/, poor/
# dataset_path = r"C:\Users\nirme\OneDrive\Desktop\project\dataset"

# # Parameters
# img_size = (128, 128)
# batch_size = 8
# num_classes = 3  # good, moderate, poor

# # Image generators
# datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# train_generator = datagen.flow_from_directory(
#     dataset_path,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training'
# )

# val_generator = datagen.flow_from_directory(
#     dataset_path,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation'
# )

# # Model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(train_generator, validation_data=val_generator, epochs=10)

# # Save the model
# model.save("hygiene_checker.keras")

# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model

# # Load model
# model = load_model("hygiene_checker.keras")
# test_image_path = r"C:\Users\nirme\OneDrive\Desktop\project\dataset\good_hygiene_images\istockphoto-1140821731-1024x1024.jpg"
# class_labels = ['good', 'moderate', 'poor']



# img = image.load_img(test_image_path, target_size=(128, 128))
# img_array = image.img_to_array(img) / 255.0
# img_array = np.expand_dims(img_array, axis=0)


# prediction = model.predict(img_array)
# class_labels = ['good', 'moderate', 'poor']

# predicted_class =class_labels[ np.argmax(prediction[0])]
    




# plt.imshow(img)
# plt.title(f"Prediction: {predicted_class}")
# plt.axis("off")
# plt.show()

# def predict_hygiene_from_frame(frame):
#     # Convert frame (OpenCV format BGR) to PIL Image
#     from PIL import Image
#     img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # Resize for model
#     img_array = image.img_to_array(img.resize((128, 128))) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction[0])
#     confidence = prediction[0][predicted_class] * 100

#     return img, class_labels[predicted_class], confidence


# # Camera capture
# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()
# cap.release()

# if ret:
#     img, predicted_class, confidence = predict_hygiene_from_frame(frame)

#     plt.imshow(img)
#     plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
#     plt.axis("off")
#     plt.show()
# else:
#     print("Camera did not capture a photo")

# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from PIL import Image

# # Path to dataset
# dataset_path = r"C:\Users\nirme\OneDrive\Desktop\project\dataset"

# # Parameters
# img_size = (128, 128)
# batch_size = 16
# num_classes = 3
# class_labels = ['good', 'moderate', 'poor']

# # Data augmentation
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# train_generator = datagen.flow_from_directory(
#     dataset_path,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training'
# )

# val_generator = datagen.flow_from_directory(
#     dataset_path,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation'
# )

# # ✅ Transfer Learning Model
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
# base_model.trainable = False

# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

# model.compile(optimizer=Adam(learning_rate=0.0001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Callbacks
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# checkpoint = ModelCheckpoint("best_hygiene_model.keras", monitor='val_accuracy', save_best_only=True)

# # Train
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=25,
#     callbacks=[early_stop, checkpoint]
# )

# # Save final model
# model.save("hygiene_checker.keras")

# # 📊 Plot training curves
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(history.history['accuracy'], label="Train Acc")
# plt.plot(history.history['val_accuracy'], label="Val Acc")
# plt.legend(); plt.title("Accuracy")

# plt.subplot(1,2,2)
# plt.plot(history.history['loss'], label="Train Loss")
# plt.plot(history.history['val_loss'], label="Val Loss")
# plt.legend(); plt.title("Loss")
# plt.show()


# # 🔎 Function to predict from single image
# def predict_image(img_path):
#     model = load_model("hygiene_checker.keras")
#     img = load_img(img_path, target_size=img_size)
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction[0])
#     confidence = prediction[0][predicted_class] * 100

#     # Print all class probabilities
#     print("\nPrediction Probabilities:")
#     for i, label in enumerate(class_labels):
#         print(f"{label}: {prediction[0][i]*100:.2f}%")

#     plt.imshow(img)
#     plt.title(f"Prediction: {class_labels[predicted_class]} ({confidence:.2f}%)")
#     plt.axis("off")
#     plt.show()


# # 🔎 Function to predict from webcam frame
# def predict_from_webcam():
#     model = load_model("hygiene_checker.keras")
#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()
#     cap.release()

#     if ret:
#         img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         img_array = img_to_array(img.resize(img_size)) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         prediction = model.predict(img_array)
#         predicted_class = np.argmax(prediction[0])
#         confidence = prediction[0][predicted_class] * 100

#         # Print all class probabilities
#         print("\nPrediction Probabilities:")
#         for i, label in enumerate(class_labels):
#             print(f"{label}: {prediction[0][i]*100:.2f}%")

#         plt.imshow(img)
#         plt.title(f"Prediction: {class_labels[predicted_class]} ({confidence:.2f}%)")
#         plt.axis("off")
#         plt.show()
#     else:
#         print("❌ Camera did not capture a photo")


# # ========================
# # ✅ Example usage:
# # ========================

# # Test on a single image
# test_image_path = r"C:\Users\nirme\OneDrive\Desktop\project\dataset\good_hygiene_images\istockphoto-1140821731-1024x1024.jpg"
# predict_image(test_image_path)

# # Test on webcam
# predict_from_webcam()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =======================
# Paths & Parameters
# =======================
dataset_path = r"C:\Users\nirme\OneDrive\Desktop\project\dataset"
img_size = (128, 128)
batch_size = 16
num_classes = 3
class_labels = ['good', 'moderate', 'poor']

# =======================
# Data Generators
# =======================
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# =======================
# Compute Class Weights
# =======================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# =======================
# Model Architecture
# =======================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# =======================
# Training
# =======================
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

model.save("hygiene_checker.keras")

# =======================
# Training Visualization
# =======================
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

plot_training_history(history)

# =======================
# Confusion Matrix
# =======================
val_generator.reset()
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)

cm = confusion_matrix(val_generator.classes, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

print("Classification Report:\n",
      classification_report(val_generator.classes, y_pred, target_names=class_labels))

# =======================
# Prediction on Single Image
# =======================
def predict_image(img_path):
    model = load_model("hygiene_checker.keras")
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100

    # Print probabilities
    print("\nPrediction Probabilities:")
    for i, label in enumerate(class_labels):
        print(f"{label}: {prediction[0][i]*100:.2f}%")

    plt.imshow(img)
    plt.title(f"Prediction: {class_labels[predicted_class]} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()

# Example usage
test_image_path = r"C:\Users\nirme\OneDrive\Desktop\project\dataset\good_hygiene_images\istockphoto-1140821731-1024x1024.jpg"
predict_image(test_image_path)

# =======================
# Prediction from Webcam
# =======================
def predict_from_webcam():
    model = load_model("hygiene_checker.keras")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_array = img_to_array(img.resize(img_size)) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100

        print("\nPrediction Probabilities:")
        for i, label in enumerate(class_labels):
            print(f"{label}: {prediction[0][i]*100:.2f}%")

        plt.imshow(img)
        plt.title(f"Prediction: {class_labels[predicted_class]} ({confidence:.2f}%)")
        plt.axis("off")
        plt.show()
    else:
        print("❌ Camera did not capture a photo")

# Example usage
# predict_from_webcam()

