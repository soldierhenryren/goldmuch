# coding=utf-8
import json
import random
import logging
import inspect
import mnist_loader

import numpy as np

# 左侧补齐15位长
FORMAT = '(P%(process)d%(processName)s-T%(thread)d%(threadName)s %(asctime)-15s>>>>)%(message)s'
# format返回logRecord的内容，
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M', level=logging.INFO)
# 获得当前模块名称的日志,不公用。
log = logging.getLogger(__name__)
class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z,a,y):
        return (a-y)

class neuralnetwork(object):
    # 参数指示使用特定方式初始化，叉熵
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        # mean 为0，方差根为1

    def feedforward(self, a):
        # zip返回的是一个迭代器生成器，每一层的输出向量，又称为向前一层的输入向量
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print "Epoch %s training complete" %j
            if monitor_training_cost:
                print "Cost on training data: {}".format(cost)
            if monitor_training_accuracy:
                print "Accuracy on training data: {} / {}".format(accuracy,n)
            if monitor_evaluation_cost:
                print "Cost on evaluation data: {}".format(cost)
            if monitor_evaluation_accuracy:
                print "Accuracy on evaluation data: {} / {}".format(self.accuracy(evalutaion_data),n_data)
            print
        return evaluation_cost,evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost.delta(zs[-1],activations[-1],y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        # 直到输入层设置完毕
        return (nabla_b, nabla_w)
    def accuracy(self,data,convert=False):
        if convert:
            results=[(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)),y) for (x,y) in data]
        return sum(int(x==y) for (x,y) in results)

    def total_cost(self,data,lmbda,convert=False):
        cost=0.0
        for x,y in data:
            a=self.feedforward(x)
            if convert:y=vectorized_result(y)
            cost += self.cost.fn(a,y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost
    def save(self, filename):
        data = {"sizes":self.sizes,
                "weights":[w.tolist() for w in self.weights],
                "biases":[b.tolist() for b in self.biases],
                "cost":str(self.cost.__name__)}
        f= open(filename,"w")
        json.dump(data,f)
        f.close()

                }
def vectorized_result(j):
    e=np.zeros((10,1))
    e[j] = 1.0
    return e

# 这是一个定义在全局范围的函数。里面是S型函数的式子。Z是一个当前层神经元的活跃值向量。Numpy的函数会自动应用到每一个元素上，得到的也是向量。
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = neuralnetwork([784, 30, 10])

net.SGD(training_data, 30, 20, 3.0, test_data=test_data)
