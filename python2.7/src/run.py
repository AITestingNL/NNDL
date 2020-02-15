import mnist_loader
import network

class Test(object):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # Network(aantal neurons (in dit voorbeeld 28 x 28 = 784 neuronen)), hiddenlayer met aantal neuronen,
    # hidden layer 2 etc, neuronen overeenkomen met 0 t/m 9)
    net = network.Network([784, 30, 20, 10, 10])

    # training data (uit mnist), aantal iteraties, mini batches, learning rate, overzicht voortgang ( x / 10000 herkend)
    net.SGD(training_data, 10, 10, 2.0, test_data=test_data)