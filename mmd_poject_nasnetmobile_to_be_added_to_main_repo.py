
import tensorflow as tf
import os
# Load the dataset
train_data = tf.data.experimental.load('gs://my-mmd/train_data')

# Print the dataset
for element,l in train_data:
    print(l.numpy().tolist())
    break

tf.config.list_physical_devices()

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
print('Running on TPU ', cluster_resolver.cluster_spec().as_dict()['worker'])

tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)

# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
# tf.config.experimental_connect_to_cluster(resolver)
# # This is the TPU initialization code that has to be at the beginning.
# tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

import tensorflow as tf
import os
# Define a callback to print metrics
class PrintMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}: {logs}")
        try:
            print(h)
        except:
            pass

strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
# Train the model
epochs = 10

tf.version.VERSION

os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'

import tensorflow as tf

with strategy.scope():
    test_data = tf.data.Dataset.load("gs://my-mmd/test_data")
    train_data = tf.data.Dataset.load("gs://my-mmd/train_data")
    val_dataset = tf.data.Dataset.load("gs://my-mmd/val_dataset")

input_shape = (256,256,3)
len(train_data)



"""# Defining the CustomNetSmall (1 attention layer) model:"""

def model_name_formatter(base_model_name):
    result = ''
    capitalize_next = True  # Flag to determine when to capitalize the character
    for char in base_model_name:
        if char.isdigit():
            result += char
            capitalize_next = True
        elif char.isalpha():
            if capitalize_next:
                result += char.upper()
                capitalize_next = False
            else:
                result += char
        else:
            result += char
    return result.replace('net','Net')

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention, Reshape, Flatten, BatchNormalization
from tensorflow.keras.applications.resnet50 import ResNet50

# Define the input shape and number of classes
input_shape = (256, 256, 3)
num_classes = 5

# Transformer Encoder Layer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = tf.add(x, inputs)

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return tf.add(x, res)

# Building the Model
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # ResNet Base
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    x = base_model.output
    x = Flatten()(x)  # Flatten the output of the convolutions
    x = Reshape((-1, x.shape[-1]))(x)  # Prepare for the Transformer

    # Transformer Encoders
    for _ in range(1):  # You can adjust the number of Transformer layers
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=256, dropout=0.1)

    # Global Average Pooling
    x = GlobalAveragePooling1D()(x)

    # Classification Head
    for units in [1024, 512, 256]:  # You can adjust the size and number of Dense layers
        x = Dense(units, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    # Create and compile the model
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Add, Flatten, Reshape, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model

# Define the input shape and number of classes
input_shape = (256, 256, 3)
num_classes = 5

# Transformer Encoder Layer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

# CNN Block
def conv_block(x, filters, kernel_size, strides=(1, 1), activation='relu'):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    return Activation(activation)(x)

# Residual Block
def residual_block(x, filters, kernel_size, strides=(1, 1)):
    shortcut = x
    x = conv_block(x, filters, kernel_size, strides)
    x = conv_block(x, filters, kernel_size, strides)
    shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    return Activation('relu')(x)

# Custom Model
def build_model(input_shape, num_classes):
    inputs = Input(input_shape)

    # Initial Convolutional Layers
    x = conv_block(inputs, 64, (7, 7), strides=(2, 2))
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Multiple Residual Blocks
    for filters in [64, 128]:
        x = residual_block(x, filters, (3, 3))
        x = residual_block(x, filters, (3, 3))

    # Flatten and reshape for transformer
    x = Flatten()(x)
    x = Reshape((-1, 64))(x)  # Reshape to (sequence_length, feature_dim)

    # Multiple Transformer Encoders
    for _ in range(2):  # Increase or decrease the range for depth
        x = transformer_encoder(x, head_size=64, num_heads=8, ff_dim=2048, dropout=0.1)

    # Global Average Pooling for sequence data
    x = GlobalAveragePooling1D()(x)

    # Dense Classifier
    for units in [1024, 512, 256]:
        x = Dense(units, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs=x)
    return model

import gc
tf.keras.backend.clear_session()
with strategy.scope():
    gc.collect()

    model = build_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Adding model name:
    model._name =  model_name_formatter("LargeCustom")

    # Model Summary
    model.summary()

"""# Training the CustomNetSmall (1 attention layer) Model:"""

import tensorflow as tf

class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=5):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_combined_metric = float('-inf')  # Initialize with a very small value
        self.wait = 0  # Counter to track the number of epochs without improvement
        self.epochs_before_stopping = 0
        self.best_weights = None  # Variable to store the best weights

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        val_accuracy = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')

        self.epochs_before_stopping += 1

        if val_accuracy is None or val_loss is None:
            print("Validation accuracy or validation loss is not found in logs. Cannot calculate custom metric.")
            return

        combined_metric = val_accuracy / (val_loss ** 2)

        if combined_metric > self.best_combined_metric:
            self.best_combined_metric = combined_metric
            self.wait = 0  # Reset the counter when a new best metric is found
            self.best_weights = self.model.get_weights()  # Save the current best weights
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Stopping training as the combined metric didn't improve for {self.patience} epochs.")
                if self.best_weights is not None:
                    print("Restoring best weights.")
                    self.model.set_weights(self.best_weights)  # Restore the best weights
                self.model.stop_training = True

import os
from tensorflow.python.profiler import profiler_client

tpu_profile_service_address = os.environ['COLAB_TPU_ADDR'].replace('8470', '8466')
print(profiler_client.monitor(tpu_profile_service_address, 100, 2))

model.summary()

with strategy.scope():
    earlyStopping = CustomEarlyStopping(patience=30)
    # Train the model
    h = model.fit(train_data,
                  epochs=500,
                  validation_data=val_dataset,
                  callbacks=[
                      # tboard_callback,
                      PrintMetricsCallback(),
                      earlyStopping
                  #     EpochModelCheckpoint("./att_model_checkpoints/epoch{epoch:02d}", frequency=1)
                  ]
              )

with strategy.scope():

    train_loss, train_accuracy = model.evaluate(train_data)
    print(f'Train accuracy: {train_accuracy}')

with strategy.scope():

    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f'Validation accuracy: {val_accuracy}')

with strategy.scope():

    test_loss, test_accuracy = model.evaluate(test_data)
    print(f'Test accuracy: {test_accuracy}')

from sklearn.metrics import classification_report, confusion_matrix
with strategy.scope():
    y_true = []
    y_pred = []

    for i,features in test_data:
        images, labels = i,features
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())

    # Compute the metrics
    accuracy = sum([1 for true, pred in zip(y_true, y_pred) if true == pred]) / len(y_true)
    loss = model.evaluate(test_data)
    classification_report = classification_report(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("Loss:", loss)

with strategy.scope():
    print("Classification Report:\n", classification_report)

with strategy.scope():
    print("Confusion Matrix:\n", confusion_mat)

import seaborn as sns
with strategy.scope():
    sns.heatmap(confusion_mat, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)

with strategy.scope():

    from sklearn.metrics import classification_report
    import pandas as pd

    clf_report = classification_report(y_true, y_pred,
                                      labels=range(5),
                                      target_names=range(5),
                                      output_dict=True)

    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)

with strategy.scope():
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot Loss
    y_loss = h.history["loss"]
    axs[0, 0].plot(y_loss, color='blue', label='Training Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Plot Accuracy
    y_accuracy = h.history["accuracy"]
    axs[0, 1].plot(y_accuracy, color='orange', label='Training Accuracy')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Plot Validation Accuracy
    y_val_accuracy = h.history["val_accuracy"]
    axs[1, 0].plot(y_val_accuracy, color='green', label='Validation Accuracy')
    axs[1, 0].set_title('Validation Accuracy')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Plot Validation Loss
    y_val_loss = h.history["val_loss"]
    axs[1, 1].plot(y_val_loss, color='red', label='Validation Loss')
    axs[1, 1].set_title('Validation Loss')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

with strategy.scope():
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot Loss
    y_loss = h.history["loss"]
    axs[0, 0].plot(y_loss)
    axs[0, 0].set_title('Loss')

    # Plot Accuracy
    y_accuracy = h.history["accuracy"]
    axs[0, 1].plot(y_accuracy)
    axs[0, 1].set_title('Accuracy')

    # Plot Validation Accuracy
    y_val_accuracy = h.history["val_accuracy"]
    axs[1, 0].plot(y_val_accuracy)
    axs[1, 0].set_title('Validation Accuracy')

    # Plot Validation Loss
    y_val_loss = h.history["val_loss"]
    axs[1, 1].plot(y_val_loss)
    axs[1, 1].set_title('Validation Loss')

    plt.tight_layout()
    plt.show()

with strategy.scope():

    import matplotlib.pyplot as plt

    y = h.history["loss"]
    plt.plot(y)
    plt.title("Loss")
    plt.show()

with strategy.scope():

    import matplotlib.pyplot as plt

    y = h.history["accuracy"]
    plt.plot(y)
    plt.title("Accuracy")
    plt.show()

with strategy.scope():

    import matplotlib.pyplot as plt

    y = h.history["val_accuracy"]
    plt.plot(y)
    plt.title("Validation Accuracy")
    plt.show()

with strategy.scope():

    import matplotlib.pyplot as plt

    y = h.history["val_loss"]
    plt.plot(y)
    plt.title("Validation Loss")
    plt.show()

"""# Sending CustomNetSmall (1 attention layer) results to GCS:"""

df = pd.DataFrame(h.history)
df

save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
try:
    f1 = clf_report['weighted avg']['f1-score']
    x = "trainable" if base_model.trainable else "not_trainable"
    df.to_csv(f"/content/my-mmd/results/tpu/{model.name}_base_model_{x}_{f1}_f1_results.csv", index=False)
    model.save(f"/content/my-mmd/results/tpu/models/{model.name}_base_model_{x}_{f1}_f1.h5", options=save_locally)
except:
    df.to_csv(f"/content/my-mmd/results/tpu/{model.name}_results.csv", index=False)
    model.save(f'/content/my-mmd/results/tpu/models/{model.name}.h5', options=save_locally)


# Sendind data to spreadseet:
with strategy.scope():

    spreadsheet_id = "1ceVqsaNC5GZ2bcduY8ULZW9NlZlcYDfS6wWhEKLOZd4"
    import gspread
    from google.auth import default
    creds, _ = default()

    c = gspread.authorize(creds)

    wb = c.open_by_key(spreadsheet_id)

    # Accessing precision, recall, and F1-score for all classes combined (last row)
    precision = clf_report['weighted avg']['precision']
    recall = clf_report['weighted avg']['recall']
    f1_score = clf_report['weighted avg']['f1-score']

    # Printing the aggregated metrics for all classes combined
    print(f"Weighted Average - Precision: {precision}, Recall: {recall}, F1-score: {f1_score}")

    # Constructing the row data to be appended
    try:
        row_data = [
            '21',  # String value
            base_model.name,  # String value
            'Yes' if base_model.trainable else 'No',
            '500',
            str(earlyStopping.epochs_before_stopping),
            str(round(float(train_accuracy),4)),  # Convert to string if not already
            str(round(float(train_loss),4)),  # Convert to string if not already
            str(round(float(val_accuracy),4)),  # Convert to string if not already
            str(round(float(val_loss),4)),  # Convert to string if not already
            str(round(float(test_accuracy),4)),  # Convert to string if not already
            str(round(float(test_loss),4)),  # Convert to string if not already
            str(round(float(precision),4)),
            str(round(float(recall),4)),
            str(round(float(f1_score),4))
        ]
    except:
        row_data = [
            '21',  # String value
            model.name,  # String value
            'Yes' if model.trainable else 'No',
            '500',
            str(earlyStopping.epochs_before_stopping),
            str(round(float(train_accuracy),4)),  # Convert to string if not already
            str(round(float(train_loss),4)),  # Convert to string if not already
            str(round(float(val_accuracy),4)),  # Convert to string if not already
            str(round(float(val_loss),4)),  # Convert to string if not already
            str(round(float(test_accuracy),4)),  # Convert to string if not already
            str(round(float(test_loss),4)),  # Convert to string if not already
            str(round(float(precision),4)),
            str(round(float(recall),4)),
            str(round(float(f1_score),4))
        ]

    ws = wb.worksheet('Sheet1')

    ws.append_row(row_data)

# If above code doesn't automatically update the values, copy the values from here and Ctrl+Shift+V on Spreadsheet to paste the values:

import math
for x in row_data:
  try:
    print(round(float(x),4), end="\t")
  except:
    print(x, end = "\t")

