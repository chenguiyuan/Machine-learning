import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

data_path = 'D:\DataRecords\Bike-Sharing-Dataset\hour.csv'
rides = pd.read_csv(data_path)
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
# print(data)
test_data = data[-21*24:]
data = data[:-21*24]
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), data[target_fields]
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]
# print(train_features)
# print(rides[:24*10].plot(x='dteday', y='cnt'))
# print(rides.head(5))
# print(train_targets['cnt'])

class NeuralNetwork(object):
   def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5, (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5, (self.hidden_nodes, self.output_nodes))
        self.activation_function = lambda x: 1/(1+np.exp(-x))

   def train(self, features, targets):
        n_records = features.shape[0]
        delta_weights_input_hidden = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_hidden_output = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_input_hidden, delta_weights_hidden_output = self.backpropagation(final_outputs, hidden_outputs, X, y, delta_weights_input_hidden, delta_weights_hidden_output)

        self.update_weights(delta_weights_input_hidden, delta_weights_hidden_output, n_records)

   def forward_pass_train(self, X):
       hidden_inputs = np.dot(X, self.weights_hidden_to_output)
       hidden_outputs = self.activation_function(hidden_inputs)

       # final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
       final_outputs = hidden_outputs

       return final_outputs, hidden_outputs

   def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_input_hidden, delta_weihts_hidden_output):

       delta_weights_input_hidden = np.zeros(self.weights_input_to_hidden.shape)
       delta_weights_hidden_output = np.zeros(self.weights_hidden_to_output.shape)
       error = y - final_outputs
       hidden_error = np.dot(self.weights_hidden_to_output, error)
       output_error_term = error
       hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)

       delta_weights_input_hidden = delta_weights_input_hidden + hidden_error_term * X
       delta_weights_hidden_output = delta_weights_hidden_output + error * hidden_outputs

       return delta_weights_input_hidden, delta_weihts_hidden_output

   def update_weights(self, delta_weights_input_hidden, delta_weights_hidden_output, n_records):
       self.weights_input_to_hidden = self.weights_input_to_hidden + delta_weights_input_hidden
       self.weights_hidden_to_output = self.weights_hidden_to_output + delta_weights_hidden_output

   def run(self, features):
       hidden_inputs = np.dot(features, self.weights_hidden_to_output)
       hidden_outputs = self.activation_function(hidden_inputs)
       # final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
       final_outputs = hidden_outputs
       return final_outputs

def MSE(self, y, Y):
       return np.mean(np.square(y - Y))


if __name__ == '__main__':
    iterations = 10
    learning_rate = 0.8
    hidden_nodes = 56
    output_nodes = 1
    # N_i = train_features.shape[1]
    input_nodes = 56
    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    losses = {'train': [], 'validation': []}
    for i in range(iterations):
        batch = np.random.choice(train_features.index, size=128)
        X = train_features.ix[batch].values
        y = train_targets.ix[batch]['cnt']
        network.train(X, y)

        # train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
        # val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
        train_loss = np.mean(np.square(network.run(train_features) - train_targets['cnt'].values))
        val_loss = np.mean(np.square(network.run(val_features) - val_targets['cnt'].values))
        # sys.stdout.write("\r Progress : {:2.1f}".format(100 * i /float(iterations))\
        #                  + "%...Training loss:" + str(train_loss[:5]\
        #                  + "%...Validation loss:" + str(val_loss[:5])))
        # sys.stdout.flush()
        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)

    print(losses)