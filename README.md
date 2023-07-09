# Sentiment Analysis Model for Game Reviews

This repository contains code and resources for training a sentiment analysis model on game reviews. The model is trained on a dataset consisting of game reviews from the "Borderlands" game series. The goal of the model is to classify the sentiment of the reviews as either positive or negative.

## Dataset

The dataset used for training the sentiment analysis model is provided in the `twitter_training.csv` file. The dataset contains the following columns:

- `id`: Unique identifier for each review
- `game`: Name of the game (in this case, "Borderlands")
- `sentiment`: Sentiment label for the review ("Positive" or "Negative")
- `text`: Text content of the review

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

2. Install the required dependencies. It is recommended to use a virtual environment:

```bash
python3 -m venv env
source env/bin/activate (Linux/Mac) or env\Scripts\activate (Windows)
pip install -r requirements.txt
```

3. Train the sentiment analysis model:

```bash
python train.py --data data.csv --model model.pth
```

This will train the model using the provided dataset and save the trained model as `model.pth` file.

4. Predict sentiment for new reviews:

```bash
python predict.py --model model.pth --text "This game is amazing!"
```

Replace the text within quotes with the actual review you want to predict the sentiment for. The predicted sentiment (positive or negative) will be displayed.

## Model Architecture

The sentiment analysis model is implemented using a deep learning approach with a recurrent neural network (RNN). The model utilizes the Long Short-Term Memory (LSTM) architecture, which has been proven effective for sequence classification tasks like sentiment analysis. The input to the model is a sequence of word embeddings, which are learned during training.

The architecture of the model consists of the following layers:

1. Embedding layer: Maps each word to a dense vector representation.
2. LSTM layer: Processes the sequence of word embeddings and captures the contextual information.
3. Fully connected layer: Performs the classification based on the output of the LSTM layer.
4. Softmax activation: Produces the final probability distribution over the two sentiment classes (positive or negative).

The model is trained using the Adam optimizer and cross-entropy loss function. The training process involves feeding batches of reviews to the model and updating the model's parameters based on the computed loss.

## Evaluation

To evaluate the performance of the sentiment analysis model, a portion of the dataset is held out as a validation set. After training the model, its performance can be assessed by calculating metrics such as accuracy, precision, recall, and F1 score on the validation set.

Additionally, it is recommended to evaluate the model on an independent test set or real-world data to get a more accurate estimate of its generalization capabilities.

## Contributions

Contributions to this repository are welcome. If you have any suggestions, bug fixes, or additional features to propose, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code for both commercial and non-commercial purposes.
