from saveModel import model

weightsFile = open('C:\\Users\Michael\\OneDrive - Pinellas County Schools\\Desktop\\SentimentAnalysis\\model_info\\model_weights.txt', 'w+')


def saveWeightsToFile(path):
    weightsFile.write(str(model.get_weights()))
    weightsFile.close()


print(model.summary())