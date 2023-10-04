import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import resample
from sklearn.svm import SVC
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score

#------------------------------------------------------------------------------------------------------------
np.random.seed(42)
SIZE=64
num_classes = 7
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
# Load metadata and image paths
skin_df = pd.read_csv('archive/HAM10000_metadata.csv')

# Encode the labels
le = LabelEncoder()
le.fit(skin_df['dx'])
LabelEncoder()
skin_df['label'] = le.transform(skin_df["dx"])
print("Skin Lesion Dataset: ")
print(skin_df['label'].value_counts())
print("\n")

# Balancing the dataset
lesion_0 = skin_df[skin_df['label'] == 0]
lesion_1 = skin_df[skin_df['label'] == 1]
lesion_2 = skin_df[skin_df['label'] == 2]
lesion_3 = skin_df[skin_df['label'] == 3]
lesion_4 = skin_df[skin_df['label'] == 4]
lesion_5 = skin_df[skin_df['label'] == 5]
lesion_6 = skin_df[skin_df['label'] == 6]

#Define the number of samples for each lesion and balance the dataset
number_samples = 900
lesion_0_balanced = resample(lesion_0, replace=True, n_samples=number_samples, random_state=42)
lesion_1_balanced = resample(lesion_1, replace=True, n_samples=number_samples, random_state=42)
lesion_2_balanced = resample(lesion_2, replace=True, n_samples=number_samples, random_state=42)
lesion_3_balanced = resample(lesion_3, replace=True, n_samples=number_samples, random_state=42)
lesion_4_balanced = resample(lesion_4, replace=True, n_samples=number_samples, random_state=42)
lesion_5_balanced = resample(lesion_5, replace=True, n_samples=number_samples, random_state=42)
lesion_6_balanced = resample(lesion_6, replace=True, n_samples=number_samples, random_state=42)

#Add data back to dataset
skin_df_balanced = pd.concat([lesion_0_balanced, lesion_1_balanced, lesion_2_balanced, lesion_3_balanced,
                              lesion_4_balanced, lesion_5_balanced, lesion_6_balanced])

#Check dataset has equal number after augmentation
print("\n")
print("Augmented Data: ")
print(skin_df_balanced['label'].value_counts())

#Read images from the CSV file (Images-> Lesion)
image_path = {os.path.splitext(os.path.basename(x))[0]: x
              for x in glob(os.path.join('archive/', '*', '*.jpg'))}
skin_df_balanced['path'] = skin_df['image_id'].map(image_path.get)
skin_df_balanced['image'] = skin_df_balanced['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))

#Convert the dataframe of images into numpy array
X = np.asarray(skin_df_balanced['image'].tolist())
X = X/225.

#Assign label values to Y and convert to categorical
Y = skin_df_balanced['label']
Y_categorical = to_categorical(Y, num_classes=7)

#Split to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y_categorical, test_size=0.25, random_state=42)

#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------

print("\n")
print("Original Dataset: ")

#Read images from the CSV file (Images-> Lesion)
image_path = {os.path.splitext(os.path.basename(x))[0]: x
              for x in glob(os.path.join('archive/', '*', '*.jpg'))}
skin_df['path'] = skin_df['image_id'].map(image_path.get)
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))

#Convert the dataframe of images into numpy array
X = np.asarray(skin_df['image'].tolist())
X = X/225.

#Assign label values to Y and convert to categorical
Y = skin_df['label']
Y_categorical = to_categorical(Y, num_classes=7)

#Split to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y_categorical, test_size=0.25, random_state=42)
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
#Exploratory Data Analysis (EDA)
class_counts = skin_df['dx'].value_counts()

# Create a bar plot of the class distribution and save as an image
plt.figure(figsize=(8,6))
plt.bar(class_counts.index, class_counts.values)
plt.title('Class Distribution')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.savefig('class_distribution.png')

# Create a histogram of the age distribution and save as an image
plt.figure(figsize=(8,6))
plt.hist(skin_df['age'], bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png')

# Print the localization distribution
print("\n")
localization_counts = skin_df['localization'].value_counts()
print(localization_counts)

# Create a bar plot of the localization distribution and save as an image
plt.figure(figsize=(22,10))
plt.bar(localization_counts.index, localization_counts.values)
plt.title('Localization Distribution')
plt.xlabel('Localization')
plt.ylabel('Count')
plt.savefig('localization_distribution.png')

# Print the sex-based distribution
print("\n")
sex_counts = skin_df['sex'].value_counts()
print(sex_counts)

# Create a pie chart of the sex-based distribution and save as an image
plt.figure(figsize=(8,6))
colors = ['blue', 'brown', 'green']
plt.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', colors=colors)
plt.title('Sex-Based Distribution')
plt.savefig('sex_distribution.png')
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
#print("\n")
print("Linear Discriminant Analysis: ")
# Perform LDA on the training data
lda = LinearDiscriminantAnalysis()
lda.fit(x_train.reshape(len(x_train), -1), np.argmax(y_train, axis=1))

# Predict on the test data
lda_y_pred = lda.predict(x_test.reshape(len(x_test), -1))

# Evaluate the performance of the classifier
print("Classification report:\n", classification_report(np.argmax(y_test, axis=1), lda_y_pred))
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
#print("\n")
print("Support Vector Machine: ")
# Create an SVM classifier
svm = SVC(kernel='poly', C=1.0)

# Train the SVM classifier
svm.fit(x_train.reshape(x_train.shape[0], -1), y_train.argmax(axis=1))

# Predict the labels of the test data
svm_y_pred = svm.predict(x_test.reshape(x_test.shape[0], -1))

# Print classification report
print("Classification report:\n", classification_report(y_test.argmax(axis=1), svm_y_pred))
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
#print("\n")
print("Convolutional Neural Network: ")
# Build the CNN Model for classification
# Define the model architecture
model = tf.keras.Sequential([
    layers.Conv2D(32, (4, 3), padding='same', activation='relu', input_shape=(SIZE, SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (4, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, (4, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(256, (4, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.25),
    layers.Dense(7, activation='softmax')
])

# Define the callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model on the training data
history = model.fit(x_train, y_train, epochs=65, batch_size=19, validation_data=(x_test, y_test), callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test data
loss, accuracy = model.evaluate(x_test, y_test)
print("Test set accuracy:", accuracy)

y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)

y_true_labels = np.argmax(y_test, axis=1)
classification_report = classification_report(y_true_labels, y_pred_labels)
print("Classification Report:\n", classification_report)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------

print("\n")
print("Ensemble Classification with CNN and SVM: ")

# Define the CNN architecture
input_shape = (SIZE, SIZE, 3)

cnn_model = Sequential()
cnn_model.add(Conv2D(32, (4, 3), activation='relu', input_shape=input_shape))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (4, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(128, (4, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(512, activation='relu'))
cnn_model.add(Dense(7, activation='softmax'))

# Compile the CNN model
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

## Define the callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

# Train the CNN model
history = cnn_model.fit(x_train, y_train, epochs=65, batch_size=19, validation_data=(x_test, y_test), callbacks=[early_stopping, reduce_lr])

# Evaluate the CNN model on the test data
loss, accuracy = cnn_model.evaluate(x_test, y_test)
print("CNN Test set accuracy:", accuracy)

# Make predictions using the CNN model
cnn_predictions = cnn_model.predict(x_test)

svm = SVC(kernel='poly', C=1.0)
svm.fit(x_train.reshape(x_train.shape[0], -1), y_train.argmax(axis=1))
svm_y_pred = svm.predict(x_test.reshape(x_test.shape[0], -1))

# Combine predictions using voting
ensemble_predictions = np.zeros_like(cnn_predictions)
for i in range(len(ensemble_predictions)):
    cnn_pred = np.argmax(cnn_predictions[i])
    svm_pred = svm_y_pred[i]
    if cnn_pred == svm_pred:
        ensemble_predictions[i][cnn_pred] = 1.0
    else:
        ensemble_predictions[i][cnn_pred] = 0.7
        ensemble_predictions[i][svm_pred] = 0.3

ensemble_accuracy = accuracy_score(y_test.argmax(axis=1), ensemble_predictions.argmax(axis=1))
print("Ensemble Accuracy:", ensemble_accuracy)
print("Classification report:\n", classification_report(y_test.argmax(axis=1), ensemble_predictions.argmax(axis=1)))