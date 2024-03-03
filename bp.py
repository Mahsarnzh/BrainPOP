import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import TFBertModel, TFGPT2Model, BertTokenizer, GPT2Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import json


# Step 1: Load and preprocess the dataset
dataset = pd.read_csv('train.csv')
# dataset = dataset.reset_index(drop=True)
# dataset = dataset.head(800)



max_len = 50
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['text'])
sequences = tokenizer.texts_to_sequences(dataset['text'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
labels = dataset['label']

# Step 2: Load BERT and GPT-2 models
bert_model = TFBertModel.from_pretrained('bert-base-uncased', trainable=True)
gpt2_model = TFGPT2Model.from_pretrained('gpt2', trainable=True)  # Set trainable=True for GPT-2

# Step 3: Define the neural net using BERT and GPT-2 models
input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32)
bert_output = bert_model(input_ids)[1]  # Using [1] to get pooled output
gpt2_output = gpt2_model(input_ids)[0]  # GPT-2 output

# Repeat BERT output for each time step in the sequence
bert_output_expanded = tf.keras.layers.RepeatVector(max_len)(bert_output)
concatenated_output = tf.keras.layers.Concatenate(axis=-1)([bert_output_expanded, gpt2_output])

# Reshape the concatenated output
reshaped_output = tf.keras.layers.Reshape((max_len, -1))(concatenated_output)

# Additional layers
lstm_layer = tf.keras.layers.LSTM(64)(reshaped_output)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(lstm_layer)

model = tf.keras.Model(inputs=input_ids, outputs=output_layer)



# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the neural net on the training set
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

X_train_np = np.array(X_train)
y_train_np = np.array(y_train)
X_test_np = np.array(X_test)
y_test_np = np.array(y_test)

# Reshape labels to match the output shape
y_train_np_categorical = y_train_np[:, np.newaxis]  # y_train_np is 1D
y_test_np_categorical = y_test_np[:, np.newaxis]  # y_test_np is 1D

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train_np, y_train_np_categorical, epochs=1, batch_size=32, validation_data=(X_test_np, y_test_np_categorical), callbacks=[early_stopping])


class Grader:
    def __init__(self, model, tokenizer, max_len):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len

    def evaluate(self, sentence, threshold=0.5):
        # Tokenize and pad the input sentence
        sequence = self.tokenizer.texts_to_sequences([sentence])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len, padding='post')

        # Predict using the model
        likelihood = self.model.predict(padded_sequence)[0][0]

        # Scale the output value between 0 and 1
        min_value = np.min(likelihood)
        max_value = np.max(likelihood)

        # Check if the range is zero
        if max_value - min_value == 0:
            scaled_value = 0.5  # Set a default value or handle as needed
        else:
            scaled_value = (likelihood - min_value) / (max_value - min_value)

        # # Convert to binary classification
        # binary_prediction = 1 if scaled_value > threshold else 0

        return likelihood

# Step 5: Create an instance of Grader
mygrader = Grader(model, tokenizer, max_len)

# Step 6: Evaluate a sample sentence
likelihood = mygrader.evaluate("she swim do she")
print(f'Likelihood that the sentence is grammatically correct: {likelihood}')


# Step 7: Evaluate the performance on the provided test set
test_data = pd.read_csv('test.csv')
test_data = test_data.reset_index(drop=True)
test_sequences = tokenizer.texts_to_sequences(test_data['text'])
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post')

predictions = model.predict(padded_test_sequences)