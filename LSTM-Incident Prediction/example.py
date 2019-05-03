import numpy
import pandas

from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 2])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('datos.csv')
dataset = dataframe.values

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be  [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], look_back, 3))
testX = numpy.reshape(testX, (testX.shape[0],look_back, 3))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back,3)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY,validation_split=0.33, nb_epoch=200, batch_size=32,
          callbacks=[callbacks.TensorBoard('logs/Graph')])

# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
#
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY, testPredict))
# print('Test Score: %.2f RMSE' % (testScore))

