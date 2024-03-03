# NLP Model Architectures

This repository contains several natural language processing (NLP) model architectures implemented using TensorFlow and Hugging Face Transformers. The models include Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), Ensemble Learning with stacking method, and a model architecture incorporating Pretrained Models (BERT and GPT-2) with CNN layers.

## Model Architectures

### 1. Simple CNN Model
- Implementation of a basic Convolutional Neural Network (CNN) for text classification.
- Achieved the best performance based on error values and Grader results.

### 2. CNN with GloVe Embedding
- CNN architecture enhanced with GloVe word embeddings for improved text representation.
In order to use glove word embeddings go to https://nlp.stanford.edu/projects/glove/ and download glove.6B.zip and find glove.6B.300d.txt in glove.6B.zip and add it to the file path, due to the files being huge I was not able to add it to github

### 3. RNN with GloVe Embedding
- Recurrent Neural Network (RNN) architecture with GloVe word embeddings for sequential data processing.
In order to use glove word embeddings go to https://nlp.stanford.edu/projects/glove/ and download glove.twitter.27B.zip and find glove.twitter.27B.200d.txt in glove.twitter.27B.zip and add it to the file path, due to the files being huge I was not able to add it to github


### 4. LSTM Model
- Implementation of a Long Short-Term Memory (LSTM) network for capturing long-term dependencies in text data.

### 5. Ensemble Learning (Stacking Method)
- Ensemble of CNN, RNN, and RandomForestClassifier using the stacking method for improved performance.

### 6. Pretrained Models (BERT and GPT-2) with CNN Layers
- Integration of pretrained BERT and GPT-2 models with a CNN architecture for advanced language understanding.

## Model Evaluation

Based on the evaluation results:
- The Simple CNN model demonstrated the best performance.
- Due to time constraints, the Pretrained Models (BERT and GPT-2) with CNN Layers were not fine-tuned.

## Usage

Clone the repository and explore the various model implementations. Adjust hyperparameters and configurations as needed.

```bash
git clone https://github.com/Mahsarnzh/BrainPOP.git
cd BrainPOP


## 6. Discuss in writing what other ideas you can suggest for improving the model:

Due to not having enough time I was not able to fine tune the models, however it is essential to emphasize that fine-tuning is a critical step for enhancing the performance of any AI model.

#### Hyperparameter Tuning:

Experiment with different hyperparameter configurations, including learning rates, batch sizes, dropout rates, and layer dimensions.
Utilize techniques such as random search or Bayesian optimization for a more systematic exploration.

#### Model Architecture:

Explore more complex architectures or try different variations of your existing models.
Consider deeper networks, additional layers, or alternative architectures like attention mechanisms.

#### Embeddings:

Utilize different pre-trained word embeddings, such as FastText, Word2Vec, or custom embeddings trained on a domain-specific corpus.
Fine-tune embeddings during training to adapt them to your specific task.

#### Data Augmentation:

Apply data augmentation techniques for text, such as synonym replacement, back translation, or paraphrasing.
Increase the diversity of your training set to improve the model's ability to handle variations in language.
