import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

dataFile = pd.read_csv("data/twitter_training.csv")

valFile = pd.read_csv('data/twitter_validation.csv')

#Normal
tweet_text = dataFile['text'].values
emotion = dataFile['sentiment'].values
#Val
tweet_val = valFile['text'].values
emotion_val = valFile['sentiment'].values



types_of_emotion = set()
types_of_emotion_val = set()
#Validation set
for emotions in emotion_val:
    types_of_emotion_val.update(emotions.split())
#Normal Set
for emotions in emotion:
    types_of_emotion.update(emotions.split())

#Normal Tokenization
types_of_emotion_updated = list(types_of_emotion)
tweet_text = np.where(pd.isnull(tweet_text), '', tweet_text)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweet_text)
tweet_seq = tokenizer.texts_to_sequences(tweet_text)

#Validation Tokenization
types_of_emotion_val = list(types_of_emotion_val)
tweet_val = np.where(pd.isnull(tweet_val), '', tweet_val)
tweet_seq_val = tokenizer.texts_to_sequences(tweet_val)

#Normal padding
max_length = max(len(seq) for seq in tweet_seq)
padding_seq = pad_sequences(tweet_seq, maxlen=max_length)

#Validation padding
max_length_val = max_length
padding_seq_val = pad_sequences(tweet_seq_val, maxlen=max_length_val)

#Normal Encoding
num_classes = len(types_of_emotion_updated)
emotion_label_dict = {label: index for index, label in enumerate(types_of_emotion_updated)}
encoded_labels = np.array([emotion_label_dict[index] for index in emotion])

#Validation encoding
num_classes_val = len(types_of_emotion_val)
emotion_label_dict_val = {label: index for index, label in enumerate(types_of_emotion_val)}
encoded_labels_val = np.array([emotion_label_dict[index] for index in emotion_val])

emotion_dict_updated = {v: k for k, v in emotion_label_dict.items()}
emotion_dict_val = {v: k for k, v in emotion_label_dict_val.items()}

types_of_emotion_updated = to_categorical(encoded_labels, num_classes=num_classes)
types_of_emotion_val = to_categorical(encoded_labels_val, num_classes=num_classes_val)

vocab_size = len(tokenizer.word_index) + 1