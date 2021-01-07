
from ear_data_set import EarDataSet
from convolutional_model import ConvolutionalModel

n_classes = 100

dataSet = EarDataSet(n_classes)

dataSet.get_data()
cnn = ConvolutionalModel(dataSet)
cnn.train(n_epochs=1000)
#cnn.evaluate()