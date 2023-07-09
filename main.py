from readData import tokenizer, pad_sequences, max_length, emotion_dict_updated
from model import model
from infoModel import saveWeightsToFile, weightsFile
import numpy as np

def predictEmotion(text:str):
    text_seq =tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(text_seq,maxlen=max_length)
    prediction = model.predict(pad_seq)[0]
    predicted_index = np.argmax(prediction, axis=-1)
    prediction_lab = emotion_dict_updated[predicted_index] 
    return prediction_lab

if __name__ == "__main__":
    input_txt = "I really like eating tic tacs"
    prediction = predictEmotion(input_txt)
    print("User: ",input_txt)
    print("Bot: ", prediction)
    saveWeightsToFile(weightsFile)