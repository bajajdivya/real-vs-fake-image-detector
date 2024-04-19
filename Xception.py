import numpy as np
import pandas as pd
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Function to plot images
def plot_images(img, label):
    plt.figure(figsize=[12, 12])
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(img[i])
        plt.axis('off')
        if label[i] == 0:
            plt.title("Fake")
        else:
            plt.title("Real")

# Function to plot training and validation history
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.show()

# Function to fit the model
def fit_model(model, base_model, epochs, fine_tune=0):
    early_stop = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)

    if fine_tune > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune]:
            layer.trainable = False
        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        base_model.trainable = False
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=valid_ds, epochs=epochs, callbacks=[early_stop], class_weight=class_weights)
    return history

# Function to create the model
def create_model(base_model):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(base_model.input, outputs)
    return model

# Paths for data
image_path = '/home/abhishek/21ucc036/real_vs_fake_faces_project/140k-real-and-fake-faces/real_vs_fake/real-vs-fake'
train_dir = image_path + '/train'
valid_dir = image_path + '/valid'
test_dir = image_path + '/test'

# Dataframe to store image paths and labels
images_df = {"folder": [],
             "image_path": [],
             "label": []}

# Iterate through folders and images to populate dataframe
for folder in ['train', 'test', 'valid']:
    for label in ['fake', 'real']:
        for img in glob.glob(f"{image_path}/{folder}/{label}/*.jpg"):
            images_df["folder"].append(folder)
            images_df["image_path"].append(img)
            images_df["label"].append(label)

images_df = pd.DataFrame(images_df)

# Image data generator
image_gen = ImageDataGenerator(rescale=1./255.)

# Data generators
train_ds = image_gen.flow_from_directory(train_dir, target_size=(256, 256), batch_size=200, class_mode='categorical')
valid_ds = image_gen.flow_from_directory(valid_dir, target_size=(256, 256), batch_size=200, class_mode='categorical')
test_ds = image_gen.flow_from_directory(test_dir, target_size=(256, 256), batch_size=200, class_mode='categorical')

# Calculate class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_ds.classes), y=train_ds.classes)
class_weights = dict(zip(np.unique(train_ds.classes), class_weights))

Xception_base_model = Xception(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
Xception_model = create_model(Xception_base_model)
Xception_model.summary()

# Number of base layers
nr_base_layers = len(Xception_base_model.layers)
print("Xception base layers = ", nr_base_layers)

# Fit the model
history = fit_model(Xception_model, Xception_base_model, epochs=5, fine_tune=int(nr_base_layers / 4))

# Save the model
Xception_model.save("Xception.h5")

# Plot training history
plot_history(history)

# Evaluate model on test data
test_loss, test_accuracy = Xception_model.evaluate(test_ds)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predictions
predictions = Xception_model.predict(test_ds)
y_pred = predictions.argmax(axis=1)

# True labels
y_true = test_ds.classes

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=test_ds.class_indices.keys(),
            yticklabels=test_ds.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification report
class_names = list(test_ds.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:")
print(report)

# Save the entire code and outputs to a PDF named 'Xception.pdf'
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('Xception.pdf') as pdf:
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=test_ds.class_indices.keys(),
                yticklabels=test_ds.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    pdf.savefig()
