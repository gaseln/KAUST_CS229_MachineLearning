import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, rms_delta=0.2, beta1=0.1, beta2=0.2):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        
        self.rms_weights_input_to_hidden = np.zeros((self.input_nodes, self.hidden_nodes))
        self.rms_weights_hidden_to_output = np.zeros((self.hidden_nodes, self.output_nodes))
        
        self.it_num = 0
        
        self.rms_delta = rms_delta
        
        self.adam_beta1 = beta1
        self.adam_beta2 = beta2
        self.adam_momentum_weights_input_to_hidden = np.zeros((self.input_nodes, self.hidden_nodes))
        self.adam_momentum_weights_hidden_to_output = np.zeros((self.hidden_nodes, self.output_nodes))
        self.adam_variance_weights_input_to_hidden = np.zeros((self.input_nodes, self.hidden_nodes))
        self.adam_variance_weights_hidden_to_output = np.zeros((self.hidden_nodes, self.output_nodes))
        
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1.0 / (1.0 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets, algorithm="grad_descent"):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        
        # this is ugly
        if (algorithm=="grad_descent"):
            self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
        elif (algorithm=="RMS_prop"):
            self.update_weights_rmsprop(delta_weights_i_h, delta_weights_h_o, n_records)
        elif (algorithm=="adam"):
            self.update_weights_adam(delta_weights_i_h, delta_weights_h_o, n_records)
            


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        output_error_term = error
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)        
        hidden_error_term = hidden_error * (1 - hidden_outputs) * hidden_outputs
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step
        
    def update_weights_rmsprop(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.rms_weights_input_to_hidden = self.rms_delta * self.rms_weights_input_to_hidden + (1 - self.rms_delta) * (delta_weights_i_h / n_records)**2
        self.rms_weights_hidden_to_output = self.rms_delta * self.rms_weights_hidden_to_output + (1 - self.rms_delta) * (delta_weights_h_o / n_records)**2
        
        self.weights_hidden_to_output += self.lr * (delta_weights_h_o / n_records) * np.sqrt(self.rms_weights_hidden_to_output)
        self.weights_input_to_hidden += self.lr * (delta_weights_i_h / n_records) * np.sqrt(self.rms_weights_input_to_hidden)
        
    
    def update_weights_adam(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.adam_momentum_weights_input_to_hidden = self.adam_beta1 * self.adam_momentum_weights_input_to_hidden + (1 - self.adam_beta1) * (delta_weights_i_h / n_records)
        self.adam_momentum_weights_hidden_to_output = self.adam_beta1 * self.adam_momentum_weights_hidden_to_output + (1 - self.adam_beta1) * (delta_weights_h_o / n_records)
        self.adam_variance_weights_input_to_hidden = self.adam_beta2 * self.adam_variance_weights_input_to_hidden + (1 - self.adam_beta2) * (delta_weights_i_h / n_records)**2
        self.adam_variance_weights_hidden_to_output = self.adam_beta2 * self.adam_variance_weights_hidden_to_output + (1 - self.adam_beta2) * (delta_weights_h_o / n_records)**2

        self.it_num += 1
        
        self.weights_hidden_to_output += self.lr * np.sqrt(1 - self.adam_beta2**self.it_num)/(1 - self.adam_beta1**self.it_num) * self.adam_momentum_weights_hidden_to_output / (np.sqrt(self.adam_variance_weights_hidden_to_output) + 0.1) 
        self.weights_input_to_hidden += self.lr * np.sqrt(1 - self.adam_beta2**self.it_num)/(1 - self.adam_beta1**self.it_num) * self.adam_momentum_weights_input_to_hidden / (np.sqrt(self.adam_variance_weights_input_to_hidden) + 0.1) 

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 2000
learning_rate = 0.2
hidden_nodes = 40
output_nodes = 1
