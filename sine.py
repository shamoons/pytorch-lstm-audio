import keras
from keras.models import Sequential, LSTM, TimeDistributedDense, Activation


model = Sequential()
model.add(LSTM(input_dim, hidden_dim, return_sequences=True))
model.add(TimeDistributedDense(hidden_dim, output_dim))
# output has shape (samples, timesteps, activities)
model.add(Activation('time_distributed_softmax'))
