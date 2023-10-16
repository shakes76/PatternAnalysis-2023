from predict import accuracyTripletModel, accuracyClassifierModel, predict
print("Predict: ")
dataLocation = './predictData'
print(predict(X=dataLocation))
print("------------------------------------------------------------------------------------")
accuracyTripletModel()
print("------------------------------------------------------------------------------------")
accuracyClassifierModel()
print("------------------------------------------------------------------------------------")
