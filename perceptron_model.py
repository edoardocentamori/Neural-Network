import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, ClassifierMixin


class Model(BaseEstimator, ClassifierMixin):
    def __init__(self, structure, eta, alpha, lamda, epochs, tolerance,activation='sigm', initialization='xav', seed=1, loss='MSE', treshold=0.5, regression=False, patience=None, early_stopping=True, new_epoch_notification=False):
        self.structure = structure  # list of perceptron units
        self.initialization = initialization
        self.seed = seed
        self.activation = activation
        self.loss = loss
        self.label_threshold = treshold
        self.regression = regression
        self.weights = []
        self.biases = []
        self.eta = eta
        self.alpha = alpha
        self.lamda = lamda
        self.patience = patience
        self.epochs = epochs
        self.losses = []
        self.accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []
        self.early_stopping = early_stopping
        self.new_epoch_notification = new_epoch_notification
        self.best_epoch = None
        self.tolerance = tolerance

    def create_model(self):
        """
        Given model with structure [a1, a2, a3, ..., ak]
        initialize parameter in the form of weights [a1 x a2, a2 x a3, ..., ak-1 x ak] and biases [1 x a2, ... 1 x ak]
        :return: weights, biases
        """
        np.random.seed(self.seed)

        # Weights creation
        weights = []
        biases = []
        if self.initialization == 'xav':  # Used for tanh
            for i in range(len(self.structure) - 1):
                weights.append(
                    np.random.randn(self.structure[i], self.structure[i + 1]) * np.sqrt(1 / self.structure[i]))
                biases.append(np.random.randn(1, self.structure[i + 1]))
        elif self.initialization == 'zero':  # Useless, it's a linear model
            for i in range(len(self.structure) - 1):
                weights.append(np.zeros((self.structure[i], self.structure[i + 1])))
                biases.append(np.zeros((1, self.structure[i + 1])))
        elif self.initialization == 'he':  # Used for ReLU
            for i in range(len(self.structure) - 1):
                weights.append(
                    np.random.randn(self.structure[i], self.structure[i + 1]) * np.sqrt(2 / self.structure[i]))
                biases.append(np.random.randn(1, self.structure[i + 1]))
        elif self.initialization == 'type1':
            for i in range(len(self.structure) - 1):
                weights.append(np.random.randn(self.structure[i], self.structure[i + 1]) * np.sqrt(
                    2 / (self.structure[i] + self.structure[i + 1])))
                biases.append(np.random.randn(1, self.structure[i + 1]))
        else:
            raise NameError("Invalid type of initialization selected")
        self.weights = weights
        self.biases = biases
        return weights, biases

    def activation_function(self, input_):
        """
        Add non-linarity to the system, you need to pass the outputed matrix to the actication function
        Allowed non-linearities are:
        sigmoid - 'sigm'
        Rectified Linear Units - 'relu'
        Hyperbolic tangent-'tanh'
        it returns the one specified by instance variable "activation".
        """
        if self.activation == 'sigm':
            return 1 / (1 + np.exp(-input_))
        elif self.activation == 'relu':
            return np.maximum(input_, 0)

        elif self.activation == 'tanh':
            return np.tanh(input_)
        else:
            raise NameError("Invalid activation function provided. Please make sure it's 'sigm' or 'relu' or 'tanh")

    def activation_derivative(self, input_):
        """
        Returns the derivative of the activation function based on parameter "activation".
        """
        if self.activation == 'sigm':
            return input_ * (1 - input_)
        elif self.activation == 'relu':
            return np.greater(input_, 0)
        elif self.activation == 'tanh':
            return 1 - input_ ** 2
        else:
            raise NameError("Invalid activation function provided. Please make sure it's 'sigm' or 'relu' or 'tanh")

    def eval_loss(self, y_true, y_pred):
        """
        :param y_true: true lables
        :param y_pred: predicted labels
        :return: loss function, either MSE or MEE
        """
        #print('%%LOSS%% {}'.format(self.loss))
        if self.loss == "MSE":
            return np.mean(np.square(y_true - y_pred))
        elif self.loss == "MME":
            return np.mean(np.sqrt(np.sum(np.square(y_true - y_pred), axis=1)))
        else:
            raise NameError("Invalid loss function provided. Please make sure it's 'MEE' or 'MSE'")

    def classify(self, input_):
        """
        Check if the values are above or below threshold, and classify them corrispondingly
        """
        #print('input_shape is {}'.format(input_.shape))
        return np.asarray([0 if i < self.label_threshold else 1 for i in input_])

    def get_accuracy(self, y_true, y_pred):
        """
        Returns the accuracy when provided the true and predicted values.
        """
        if not self.regression:
            y_pred = self.classify(y_pred)
            #print('pred: {} true {}'.format(y_pred.shape, y_true[:, 0].shape))
            #print(y_pred)
            #print(y_true)
        return accuracy_score(y_pred=y_pred, y_true=y_true[:, 0])  #DEBUG added [0]

    def score_training(self, y_true, y_pred, acc=False, loss=False):
        """
        :param y_true: true labels
        :param y_pred: predicted labels
        :param acc: flag, if true return accuracy
        :param loss: flag, if true return loss
        """
        if acc and loss:
            return self.get_accuracy(y_true, y_pred), self.eval_loss(y_true, y_pred)
        elif acc:
            return self.get_accuracy(y_true, y_pred)
        elif loss:
            return self.eval_loss(y_true, y_pred)
        else:
            raise ValueError("Please specify what to score, 'acc' for accuracy, 'loss' for loss. Set them to True")

    def forward_prop(self, input_datas):
        """
        :param input_datas: matrix of datas
        :return: output of every layer but the first.
        """
        flow_history = []
        flowing_data = input_datas
        for i, W in enumerate(self.weights):
            linear_next = np.dot(flowing_data, W) + self.biases[i]  # W*X+b
            flowing_data = self.activation_function(linear_next)  # Non linearity
            if i == len(self.weights) - 1 and self.regression:
                flow_history.append(linear_next)
            else:
                flow_history.append(flowing_data)
        return flow_history

    def back_prop_intralayer(self, previous_delta, current_out, weight_c_p, is_output_layer=False, label_matrix=None):
        """
        previous layer is the one more close to output just above the current one
        :param previous_delta: delta evaluated in the previous layer
        :param current_out: output of this layer in the forward prop
        :param weight_c_p: weights connecting this layer with the previous layer
        :param is_output_layer: flag, if true deal with differences related to output
        :param label_matrix: use only if is_output_layer, in that case give real label matrix
        :return: New evaluation of delta for this layer
        """
        deriv = self.activation_derivative(current_out)
        if not is_output_layer:
            new_delta = previous_delta.dot(weight_c_p.T) * deriv
        else:
            difference = label_matrix - current_out
            if self.regression:
                new_delta = normalize(difference, axis=1,
                                          norm='l1')  # output doesn't passes through nonlinear function for regression
            else:
                new_delta = difference * deriv
        return new_delta

    def back_prop(self, label_matrix, forward_out):
        """
        :param label_matrix:
        :param forward_out:
        :return:
        """
        reversed_delta_history = []
        label_out = forward_out[-1]
        # Output layer back prop
        out_delta = self.back_prop_intralayer(None, label_out, None, is_output_layer=True, label_matrix=label_matrix)
        # Everything else back prop
        backflowing_delta = out_delta
        reversed_delta_history.append(backflowing_delta)
        for i in range(len(self.structure)-2):
            current_out = forward_out[-2-i]
            weight_c_p = self.weights[-1-i]
            backflowing_delta = self.back_prop_intralayer(backflowing_delta, current_out, weight_c_p, is_output_layer=False, label_matrix=None)
            reversed_delta_history.append(backflowing_delta)
        return reversed_delta_history[::-1]

    def update_weights(self, data_matrix, forward_out, delta_history, prev_delta_W):
        """
        :param data_matrix: input matrix
        :param forward_out: output of forward_prop
        :param delta_history: output of back_prop
        :param prev_delta_W: previous output of update_weights (if exist, otherwise pass 0)
        :return: updating values for weights
        """
        delta_W=[]
        other_delta_W=[]
        delta_W.append(data_matrix.T.dot(delta_history[0]) * self.eta)
        other_delta_W.append(self.eta * self.weights[0] * (-self.lamda) + self.alpha * prev_delta_W[0])
        for i in range(len(forward_out)-1):
            delta_W.append(forward_out[i].T.dot(delta_history[i+1]) * self.eta)
            other_delta_W.append(self.eta * self.weights[i+1] * (-self.lamda) + self.alpha * prev_delta_W[i+1])
        if self.regression:
            for i in range(len(delta_W)):
                delta_W[i] = delta_W[i] / data_matrix.shape[0]
        for i in range(len(delta_W)):
            self.weights[i] += delta_W[i] + other_delta_W[i]
        for i in range(len(self.biases)):
            self.biases[i] += np.sum(delta_history[i], axis=0, keepdims=True) * self.eta
        if self.regression:
            for i in range(len(self.biases)):
                self.weights[i] /= data_matrix.shape[0]
        return delta_W

    def predict(self, data_matrix, labels=None, acc_=False, from_val=False):
        """
        predict the test/validation dataset
        to get accuracy as well, set acc_ to true
        to get the losses, pass true labels and result of this function to getLoss() function
        """
        result = self.forward_prop(data_matrix)[-1]
        if acc_:
            accuracies = []
            for i in range(data_matrix.shape[0]):
                assert labels is not None, "true values (as labels) must be provided for to calculate accuracy"
                accuracies.append(self.score_training(labels[i], result[i], acc=True))
            return result, np.sum(accuracies)/data_matrix.shape[0]
        if from_val or self.regression:
            return result
        else:
            return self.classify(result)

    def fit(self, features, labels, validation_features=None, validation_labels=None, early_stopping_log=True, coming_from_grid_search=False):
        """
        Train the model
        :param features:
        :param labels:
        :param validation_features:
        :param validation_labels:
        :param early_stopping_log:
        :param coming_from_grid_search:
        :return:
        """
        weights, biases = self.create_model()
        self.accuracies = []
        self.losses = []
        if validation_features is not None:
            self.validation_accuracies = []
            self.validation_losses = []
        delta_W = [0] * len(self.weights)  # fix conflicts
        for i, weight in enumerate(self.weights):
            delta_W[i] = np.zeros(weight.shape)
        patience = self.patience
        for iteration in range(self.epochs):
            print("iteration {}/{}".format(iteration + 1, self.epochs), end="\r")
            forward_out = self.forward_prop(features)
            y_pred = forward_out[-1]
            deltas = self.back_prop(labels, forward_out)
            prev_delta_W = delta_W
            delta_W = self.update_weights(features, forward_out, deltas, prev_delta_W)
            epoch_loss = self.score_training(labels, y_pred, loss=True)
            self.losses.append(epoch_loss)
            if not self.regression:
                epoch_accuracy = self.score_training(labels, y_pred, acc=True)
                self.accuracies.append(epoch_accuracy)
            if validation_features is not None:
                assert len(validation_features) == len(validation_labels), "validation features and labels length shoud be the same"
                validation_results = self.predict(validation_features, acc_=False, from_val=True)
                self.validation_losses.append(self.score_training(validation_labels, validation_results, loss=True))
                if not self.regression:
                    self.validation_accuracies.append(self.score_training(validation_labels, validation_results, acc=True))
            if self.early_stopping:
                if iteration > 0:
                    if coming_from_grid_search:
                        self.new_epoch_notification = False
                    loss_decrement = (self.losses[iteration - 1] - self.losses[iteration]) / self.losses[iteration - 1]
                    if loss_decrement < self.tolerance:
                        patience -= 1
                        if patience == 0:
                            if early_stopping_log:
                                print("The algorithm has run out of patience. \nFinishing due to early stopping on epoch {}. \n PS. Try decreasing 'tolerance' or increasing 'patience'".format(iteration))
                            self.new_epoch_notification = True
                            self.best_epoch = iteration
                            break
                    else:
                        patience = self.patience



'''
A = Model([3, 17, 5, 1], 0.005, 0.01, 0.01, 2000, 'sigm', 'xav', 3, 'MSE', 0.5, False, False, False, True)
eights, biases = A.create_model()
x_train = np.random.rand(100, 3)
true_labels = np.random.rand(100, 1)
results = A.forward_prop(x_train)

fake_prev_delta_W = []
for weight in A.weights:
    print(weight.shape[0], weight.shape[1])
    fake_prev_delta_W.append(np.random.rand(weight.shape[0], weight.shape[1]))


print('-------')
print('FORWARD')
for line in results:
    print(line.shape)

print('--------')
print('DELTAS')
deltas = A.back_prop(true_labels, results)
for delta in deltas:
    print(delta.shape)
print('--------')
deltaW = A.update_weights(x_train, results, deltas, fake_prev_delta_W)
for deltaw in deltaW:
    print(deltaw.shape)
    #print(deltaw)
print('--------')

x_train = np.random.rand(5000, 3)
x_val = np.random.rand(2000, 3)
y_intermediate = x_train[:, 1]-x_train[:, 0]**2
y_val_int = x_val[:, 1]-x_val[:, 0]**2
y_train = np.asarray([[1 if i > 0.5 else 0 for i in y_intermediate]]).T
y_val = np.asarray([[1 if i > 0.5 else 0 for i in y_val_int]]).T
#print('shapes x {}, y {}'.format(x_train.shape, y_train.shape))
#y_train = np.array([x_train[:, 1]-x_train[:, 0]**2]).T
print('Before fit')
print(x_train.shape, y_train.shape)

A.fit(x_train, y_train, validation_features=x_val, validation_labels=y_val, early_stopping_log=True, coming_from_grid_search=False)
print('DONE!!')

import matplotlib.pyplot as plt

X = np.arange(len(A.losses))

fig, ax = plt.subplots()
ax.plot(X, A.losses, c='b')
ax.plot(X, A.validation_losses, c='r')
plt.show()
'''