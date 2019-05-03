import numpy as np
import pandas as pd
import time
import tensorflow as tf
import rpy2.robjects as ro

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from keras import backend as K
from imblearn.combine import SMOTEENN

def f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='weighted')
    return f1

# f1 calculation after end of each epochs
class Metrics(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.f1 = []

    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        self.f1.append(f1_score(targ, predict, average='weighted'))
        return

# define auc metric
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 26])
    return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
df = pd.read_csv('data\\2016-2017_incident_data.csv')
df = df.drop(columns=['DateTime','Location','Direction','Cause','S13Count','S14Count','S15Count'])

# normalize the dataset
scaler = MinMaxScaler(feature_range=(1, 2))
dataset = pd.DataFrame(scaler.fit_transform(df))

# split into train and test sets
train_size = int(len(dataset) * 0.67)
train, test = dataset.iloc[0:train_size, :], dataset.loc[train_size:len(dataset), :]

# # resample using SMOTE + ENN (combine of over and under sampling)
# sm = SMOTEENN()
# X = train.iloc[:, 0:-1]
# y = train.iloc[:, -1]
# X_resample, y_resample = sm.fit_sample(X, y)
# X_resample, y_resample = pd.DataFrame(X_resample), pd.DataFrame(y_resample)
# train = pd.merge(X_resample, y_resample, left_index=True, right_index=True)

# over sample using EPSO and ADASYN from r package
pandas2ri.activate()
ostsc = importr("OSTSC")
X, y = train.iloc[:, 0:-1], train.iloc[:, -1]
# replace all 0 with non zero values to avoid calculation error when sampling
# X = X.replace(0, 0.001)
r_X, r_y = pandas2ri.py2ri(X), pandas2ri.py2ri(pd.DataFrame(y))
resample = ostsc.OSTSC(r_X, r_y)
train = pd.merge(resample.sample, resample.label, left_index=True, right_index=True)

# set basic parameter
look_back = 26
features = 25
epochs = 300
batch_size = 1024

# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train.values, look_back)
testX, testY = create_dataset(test.values, look_back)

# reshape input to be  [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], look_back, features))
testX = np.reshape(testX, (testX.shape[0],look_back, features))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(16, input_shape=(look_back,features)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])
print(model.summary())

start_time = time.time()
metrics = Metrics()
model.fit(trainX, trainY, validation_split=0.33, epochs=epochs, batch_size=batch_size,
          callbacks=[callbacks.TensorBoard('logs/Graph'), metrics])
                     #,callbacks.EarlyStopping(patience=10)])
end_time = time.time()
print("--- runtime --- %s seconds" % (end_time-start_time))

# make predictions
test_Pred = model.predict(testX)
# calculate loss, metric on test set
score = model.evaluate(testX, testY, batch_size=16)
# confusion matrix
pd.crosstab(testY, test_Pred[:,0], rownames=['Actual'], colnames=['Predicted'])