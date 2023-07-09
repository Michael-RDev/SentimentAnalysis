from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Activation 
from keras.callbacks import EarlyStopping
import numpy as np

from readData import num_classes, max_length, padding_seq_val, types_of_emotion_val, padding_seq, types_of_emotion_updated, vocab_size


stawp_val = EarlyStopping('val_loss', verbose=3)
stawp_acc = EarlyStopping('val_accuracy', verbose=3)

callBacksAvaliable = [stawp_val, stawp_acc]



def predictionModel(input_shape:int):
    model = Sequential()
    model.add(Embedding(input_dim=input_shape, output_dim=100))
    model.add(LSTM(120))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


model = predictionModel(input_shape=vocab_size)

epochs = 2

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(padding_seq, np.argmax(types_of_emotion_updated, axis=-1), epochs=epochs, callbacks=callBacksAvaliable, validation_data=(padding_seq_val, np.argmax(types_of_emotion_val, axis=-1)))