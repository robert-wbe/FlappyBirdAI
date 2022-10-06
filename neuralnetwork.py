import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        t = sigmoid(x)
        return t * (1-t)
    return 1/(1+np.exp(-x))

def relu(x, deriv=False):
    if deriv:
        if x > 0:
            return 1
        return 0
    if x > 0:
        return x
    return 0

def lrelu(x, deriv=False):
    if deriv:
        if x > 0.05:
            return 1
        return 0.05
    if x > 0.05:
        return x
    return 0.05

def linear(x, deriv=False):
    if deriv:
        return 1
    return x

def ls(list):
    temp = ""
    for i in list:
        temp += str(i) + ":"
    return temp[:-1]

def listadd(l1,l2):
    return [i+j for i,j in zip(l1,l2)]

def listsubtract(l1,l2):
    return [i-j  for i,j in zip(l1,l2)]

def listdivide(l,d):
    return [i/d for i in l]

def listmultiply(l,f):
    return[i*f for i in l]


class layer:
    def __init__(self, *args, afunc = sigmoid):

        self.afunc = afunc
        # Assignment of the activation function
        if len(args):
            # Set random weights and biases

            self.weights = 0.1*np.random.randn(args[1], args[0])
            # args: n_inputs, n_neurons
            # shape := n_neurons x [w(1), w(2), ..., w(n_inputs)] (numpy matrix)

            self.biases = np.zeros(args[1])
            # shape := [b(1), b(2), ..., b(n_neurons)]

    def forward(self, inputs):
        # FeedForward
        self.z = np.dot(self.weights, inputs) + self.biases
        self.outputs = np.array([self.afunc.__call__(x) for x in self.z])
        # Activation function is applied to every component of the dot product
        # Dot product of weights and inputs: For every neuron the dot product of its weights and the inputs


class NeuralNetwork:
    def __init__(self, *args, afunc=sigmoid):
        self.layers = []

        if len(args):
            self.size = len(args)

            # Initiate layers randomly
            for inp, out in zip(args[:-1], args[1:]):
                self.layers.append(layer(inp, out, afunc=afunc))
            # shape := [layer(s1,s2), layer(s2,s3), ..., layer(sn-1,sn)]
            self.update_w_and_b()

    def export(self, local=False):
        # If 'local' is stated, returns the content string in the console
        # If 'local' is not stated, writes the content string to a file


        string = str(self.size) + "\n"
        for i in self.layers:
            for j in i.weights:
                string += ls(j) + ","
            string += ls(i.biases) + ";"
        if local:
            print(string[:-1])
        else:
            data = open("data.txt", "w")
            data.write(string[:-1])
            data.close()

    def setImport(self, data):
        # Sets the networks weights and biases according to a content string, while adapting the shape

        self.size = int(data.split("\n")[0])
        self.layers = [layer() for i in range(self.size-1)]
        w_and_b = data.split("\n")[1].split(";")
        for l in range(self.size-1):

            self.layers[l].weights = np.array([[float(w) for w in n.split(":")] for n in w_and_b[l].split(",")[:-1]])
            self.layers[l].biases = np.array([float(b) for b in w_and_b[l].split(",")[-1].split(":")])
        self.update_w_and_b()

        # +++NOW THE IMPORTANT PART BEGINS!+++
        # +++NOW THE IMPORTANT PART BEGINS!+++

    def run(self, inputs):
        self.netinputs = np.array(inputs)
        temp = inputs
        # Apply each layer
        for i in self.layers:
            i.forward(temp)
            temp = i.outputs
        return temp

    def update_w_and_b(self):
        # Updates all the weights and biases from layer attributes to Network attributes.

        self.weights = [l.weights for l in self.layers]
        self.biases = [l.biases for l in self.layers]

    def update_layers(self):
        for l,i in zip(self.layers,range(len(self.layers))):
            l.weights = self.weights[i]
            l.biases = self.biases[i]
    
    def mutate(self,amount):
        "use for neuroevolution"
        new = NeuralNetwork()
        new.size = self.size
        new.weights = listadd(self.weights, [np.random.randn(len(self.weights[i]),len(self.weights[i][0]))*amount for i in range(new.size-1)])
        new.biases = listadd(self.biases, [np.random.randn(len(self.biases[i]))*amount for i in range(new.size-1)])
        for i in range(self.size-1):
            new.layers.append(layer())
        new.update_layers()
        return new

    def backprop(self, inputs, exp):
        # This core method computes the delta to the GRADIENT DESCENT caused by a single example.
        # Get the outputs
        outputs = self.run(inputs)
        expected = np.array(exp)
        self.error = outputs-expected
        
        activations = [l.outputs for l in self.layers[-2::-1]] + [self.netinputs]

        # Calculate the gradient for the layer's activation
        delta_layer_z = 2*(outputs - expected) * np.array([self.layers[-1].afunc.__call__(i,deriv = True) for i in self.layers[-1].z])

        # Calculate the gradient compnent for the weghts and biases for this example
        delta_w = []
        delta_b = []

        for layer, pac in zip(self.layers[::-1],activations):
            l_delta_w = np.array([pac * delta_layer_z[neuron] for neuron in range(layer.weights.shape[0])])
            # This is the influence of the layer's weights on the error function.
            delta_w.append(l_delta_w)

            l_delta_b = delta_layer_z
            # This is the influence of the layer's biases on the error function.
            delta_b.append(l_delta_b)

            delta_layer_z = np.dot(layer.weights.transpose(),np.array([layer.afunc.__call__(a,deriv=True) for a in layer.z])*delta_layer_z)
            # This is the influence of the previous layer's activation on the error function.

        # Return the two gradient components
        return delta_w[::-1], delta_b[::-1]

    def update_mini_batch(self, mini_batch, lr):
        # Update the network's weights and biases by applying
        # gradient descent using backpropagation to a single mini batch.
        # The 'mini batch' is a list of tuples (of numpy arrays) '(inp, exp)', and 'lr' is the learning rate

        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        # Together, these form the "gradient descent" of the Neural Network

        for i, e in mini_batch:
            # Get the gradient component of a single example
            d_gradient = self.backprop(i,e)

            # Add them to the final gradient descent
            gradient_w = listadd(d_gradient[0],gradient_w)
            gradient_b = listadd(d_gradient[1],gradient_b)

        # Divide by the batch size to get the average for the gradient.
        #print(self.error)
        gradient_b = listdivide(gradient_b, len(mini_batch))
        gradient_w = listdivide(gradient_w, len(mini_batch))

        # Change the networks weights and biases by the NEGATIVE GRADIENT (GRADIENT DESCENT) multiplied by the learning rate

        self.weights = listsubtract(self.weights, listmultiply(gradient_w,lr))
        self.biases = listsubtract(self.biases, listmultiply(gradient_b,lr))
        self.update_layers()
