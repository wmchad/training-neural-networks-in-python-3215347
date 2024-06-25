import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.weights = (np.random.rand(inputs+1) * 2) - 1 
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x,self.bias),self.weights)
        return self.sigmoid(x_sum)

    def set_weights(self, w_init):
        """Set the weights. w_init is a python list with the weights."""
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """Evaluate the sigmoid function for the floating point input x."""
        return 1/(1+np.exp(-x))



class MultiLayerPerceptron:     
    """A multilayer perceptron class that uses the Perceptron class above.
       Attributes:
          layers:  A python list with the number of elements per layer.
          bias:    The bias term. The same bias is used for all neurons.
          eta:     The learning rate."""

    def __init__(self, layers, bias = 1.0):
        """Return a new MLP object with the specified parameters.""" 
        self.layers = np.array(layers,dtype=object)
        self.bias = bias
        self.network = [] # The list of lists of neurons
        self.values = []  # The list of lists of output values

        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:      #network[0] is the input layer, so it has no neurons
                for j in range(self.layers[i]): 
                    self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias))
        
        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values = np.array([np.array(x) for x in self.values],dtype=object)

# Challenge: Finish the set_weights() and run() methods:

    def set_weights(self, w_init):
        # Write all the weights into the neural network.
        # w_init is a list of floats. Organize it as you'd like. 
        for i in range(1, len(self.layers)):
            for j in range(self.layers[i]):
                self.network[i][j].set_weights(w_init[i - 1][j])

    def print_weights(self):
        print()
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                print("Layer", i+1, "Neuron", j, self.network[i][j].weights)
        print()

    def run(self, x):
        # Run an input forward through the neural network.
        # x is a python list with the input values.
        self.values[0] = np.array(x, dtype = object)
        for i in range(1, len(self.network)):
            self.values[i] = np.array([p.run(self.values[i - 1]) for p in self.network[i]])
        return self.values[-1]
        
mlp = MultiLayerPerceptron([2, 2, 1])
mlp.print_weights()
mlp.set_weights([
    [[-10, -10, 15], [15, 15, -10]],
    [[10, 10, -15]]
    ])
mlp.print_weights()
print("0, 0: ", mlp.run([0, 0]))
print("0, 1: ", mlp.run([0, 1]))
print("1, 0: ", mlp.run([1, 0]))
print("1, 1: ", mlp.run([1, 1]))