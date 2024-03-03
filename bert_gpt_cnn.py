import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import TFBertModel, TFGPT2Model, BertTokenizer, GPT2Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout


# Step 1: Load and preprocess the dataset
dataset = pd.read_csv('train.csv')
# dataset = dataset.reset_index(drop=True)
dataset = dataset.head(800)


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

# Flatten the reshaped output to make it compatible with the CNN model
# flattened_output = tf.keras.layers.Flatten()(reshaped_output)

embedding_dim = 128
cnn_model = Sequential()
cnn_model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_len))

# Add Conv1D directly to cnn_model, removing the sequential part
cnn_model.add(Conv1D(128, 5, activation='relu'))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(1, activation='sigmoid'))

# No need to call cnn_model on reshaped_output, just connect the layers
output_cnn = cnn_model(reshaped_output)

# Define the final model
model = tf.keras.Model(inputs=input_ids, outputs=output_cnn)


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
