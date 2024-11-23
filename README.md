# CNN_TF
This code trains a Convolutional Neural Network (CNN) using the MNIST dataset of handwritten digits, evaluates its performance, and visualizes the predictions. Here's a detailed breakdown of the code:
________________________________________
1. Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
•	tensorflow: The framework used for building and training neural networks.
•	layers and models: Submodules in Keras for defining the CNN architecture.
•	mnist: A dataset provided by Keras, containing 70,000 images of handwritten digits (60,000 for training and 10,000 for testing).
________________________________________
2. Load and Prepare Data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
•	Loads the MNIST dataset into training and testing splits.
•	train_images: Images for training the model.
•	train_labels: Corresponding digit labels for training images.
•	test_images and test_labels: Reserved for model evaluation.
________________________________________
2.1 Normalize the Pixel Values
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
•	Scales pixel values from 0-255 to 0-1 to make training faster and improve model performance.
________________________________________
2.2 Reshape Images
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
•	Reshapes the data to include a channel dimension (28x28x1).
•	CNNs require input with explicit channel dimensions (e.g., grayscale images have 1 channel).
________________________________________
2.3 One-Hot Encode Labels
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
•	Converts labels (0–9) into one-hot vectors. Example: 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
•	This format is required for the categorical cross-entropy loss function.
________________________________________
3. Define the CNN Model
model = models.Sequential()
•	Sequential: Used to define a linear stack of layers.
Add Layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
•	Convolutional layers: 
o	Extract features from the input image using filters (3x3 in size).
o	First layer has 32 filters, subsequent layers have 64 filters.
o	ReLU activation function introduces non-linearity.
•	MaxPooling layers: 
o	Down-sample the image dimensions by taking the max value in 2x2 regions, reducing spatial size and computations.
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
•	Flatten: Converts the 2D feature maps into a 1D vector.
•	Dense Layers: 
o	Fully connected layers. First layer has 64 neurons.
o	Final layer has 10 neurons (for 10 digit classes) and uses softmax for probabilistic outputs.
________________________________________
4. Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
•	Optimizer: Adam (Adaptive Moment Estimation) adjusts learning rates dynamically.
•	Loss: Categorical cross-entropy measures the difference between predicted and true distributions.
•	Metrics: Accuracy evaluates model performance.
________________________________________
5. Train the Model
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)
•	Training data: train_images and train_labels.
•	Epochs: Number of passes through the entire training dataset.
•	Batch size: Number of samples per gradient update.
•	Validation split: Reserves 20% of training data for validation.
________________________________________
6. Evaluate the Model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
•	Measures the loss and accuracy on the test dataset.
________________________________________
7. Make Predictions
predictions = model.predict(test_images)
predicted_labels = predictions.argmax(axis=1)
•	predict: Produces probabilities for each class.
•	argmax: Selects the class with the highest probability for each prediction.
________________________________________
8. Visualize Predictions
Define Visualization Function
def plot_images(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i].argmax(), img[i].reshape(28, 28)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = predictions_array.argmax()
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"Predicted: {predicted_label} ({100 * tf.reduce_max(predictions_array):.2f}%), Actual: {true_label}", color=color)
•	Displays: 
o	The image.
o	Predicted label and probability.
o	True label.
•	Label color: Blue if correct, red if incorrect.
Plot Predictions
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_images(i, predictions, test_labels, test_images)
plt.show()
•	Creates a grid of images and their predictions.
•	Blue labels indicate correct predictions; red labels indicate incorrect predictions.
________________________________________
Summary
1.	The model is trained on MNIST data to classify handwritten digits.
2.	After training, the model is evaluated for accuracy.
3.	Predictions are visualized to compare actual and predicted labels.
“This process demonstrates how CNNs can be used for image classification tasks.”

