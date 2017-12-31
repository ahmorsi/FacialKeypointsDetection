from keras.losses import mean_squared_error
from keras import optimizers
from util import load
from Models import SimpleCNN
import numpy as np

batch_size = 128
epochs = 25

print('Reading Train Data')
X, y,cols_names = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))

feature_name2KeypointIdx = {}
for idx,feature_name in enumerate(cols_names):
    feature_name2KeypointIdx[feature_name] = idx

np.save('feature2kpId.npy',feature_name2KeypointIdx)

model = SimpleCNN()
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
print('Start Training')
model.compile(loss=mean_squared_error,
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X, y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

print('Done Training')
print('Saving Weights')
model.save_weights("FKD_weights.h5")