# NLP Model Architectures

This repository contains several natural language processing (NLP) model architectures implemented using TensorFlow and Hugging Face Transformers. The models include Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), Ensemble Learning with stacking method, and a model architecture incorporating Pretrained Models (BERT and GPT-2) with CNN layers.

## Model Architectures

### 1. Simple CNN Model
- Implementation of a basic Convolutional Neural Network (CNN) for text classification.
- Achieved the best performance based on error values and Grader results.

### 2. CNN with GloVe Embedding
- CNN architecture enhanced with GloVe word embeddings for improved text representation.

### 3. RNN with GloVe Embedding
- Recurrent Neural Network (RNN) architecture with GloVe word embeddings for sequential data processing.

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
git clone https://github.com/yourusername/your-repository.git
cd your-repository
