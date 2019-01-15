# coding=utf-8
import random
import logging
import mnist_loader

import numpy as np

# # 左侧补齐15位长
# FORMAT = '(P%(process)d%(processName)s-T%(thread)d%(threadName)s %(asctime)-15s>>>>)%(message)s'
# # format返回logRecord的内容，
# logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M', level=logging.INFO)
# # 获得当前模块名称的日志,不公用。
# log = logging.getLogger(__name__)
# python解释器在用到这个neuralnetwork类的实例的时候，就会新建并初始化它，这里是定义这个类是怎样的存在。
# 这个类继承自object类，是所有新样式类的基类。(类型type和类class的统一，built-in 类型和用户自定义类之间的差别消除，禁止list 类型和 dictionaries类型做类语法的基类，广度优先搜索，经典类深度优先。)
class neuralnetwork(object):
    # 这是object类的静态方法 static method，cls是这个neuralnetwork类对象。后面的参数和，关键字参数是调用类时传入的，星号表示数量任意。返回值就是cls的实例啦。
    def __new__(cls, *args, **kwargs):
        # 实际上会调用父类object的
        return super(neuralnetwork, cls).__new__(cls)

    # __init__在__new__类的实例被创建之后执行,此函数结束后，实例被返回给调用者
    # sizes是列表类型，分别指定第1层神经元数量，第2层神经元数量，依次类推，项数指定层数，每项的数值表示该层的神经元数量。
    def __init__(self, sizes):
        # 神经网络的层的数量
        self.num_layers = len(sizes)
        # 神经网络神经元数量列表。
        self.sizes = sizes
        # sizes[1:] sizes是一个list类型，bracket符号方法获取一个sequences slicing切片，colon 前面的是开始项，后面省略表示len（sizes）
        # 整个表达式是列表的生成式list comprehension 不是生成器表达式 generator expressions（一个生成列表，一个生成生成器迭代器）
        # np包,random子包里的，mtrand模块里RandomState类的randn方法，但是在random子包的__init__.py中使用__all__列表限制了import能导入的特征。这是一个内建
        # 方法，mtrand模块和RandomState类是不可见的。但是可以用下面的方法找到，不过，这样找到的是一个方法对象，并不是内建函数，
        # logging.info(np.random.__dict__['mtrand'].RandomState.__dict__['randn'])
        # 而且这个方法对象是method_descriptor，所以肯定是在__get__上做了手脚，使得获得的实际上是一个内建函数。
        # logging.info(inspect.getmembers(np.random.__dict__['mtrand'].RandomState.__dict__['randn']))
        # 不过我们关注的重点还是2个论数，表示一个2阶量，y个维度上，每个维度有n维数据（此处是1）。这是一个y行1列的矩阵，生成器yield了，1到y维。
        # 得到的是每层神经元们都生成一个多行1列的矩阵，也是列向量，表示每一层神经元的偏移值向量（随机取值，符合正态分布）,输入层没有偏移量。
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # y维度上有x维的数据。每层神经元都有一个矩阵，矩阵的行表示相应的神经元，矩阵的列表示相应神经元的输入权重。
        # zip使用两个论数，每次等序列的各取一个使用，从神经网络的第二层的第0个神经元从第一层神经元的数量数就是输入权重，一共有列表的上一项数字个
        # （代表上一层的神经元数量）x个，一直到最后一个，第二层有y个神经元。列表生成器，每一轮生成一个字典，该层神经元数y为行数，每个神经元来自
        # 上一层神经元数量，表示的输出数量，x，生成一个矩阵。总的结果就是矩阵组成的列表。列表数是神经网络层的数量。
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        # mean 为0，方差根为1

    # 方法使用a作为输入，这是一个1列的数组,对于从后向前的每一层神经网络,都要取出b偏移值向量，w权重矩阵
    def feedforward(self, a):
        # zip返回的是一个迭代器生成器，每一层的输出向量，又称为向前一层的输入向量
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # 训练数据是训练输入和渴望的输出的团(x,y)组成的列表。其他费可选参数，比如正则化参数regularization parameter lmbda（提高拟合度，
    # 防治过拟合。也接受修正用数据，要么是校验数据，要么是测试数据。我们能监控代价和精度在每一个修正数据或者训练数据，通过设置合适的
    # 标记。这个方法返回包含四个列表的团：（每轮）在修正数据上的代价，修正数据上的精准度，训练数据的代价，训练数据的精准度。所有的值被修正
    # 在每一个训练轮的结束。所有，例如，如果我们训练30轮，那么团的第一个元素就是一个30元素的包含每轮结束时修正数据的代价。注意如果相应的
    # 标记没有设置，列表为空）。
    # def SGD(self,training_data, epochs, mini_batch_size, eta,
    #         lmbda = 0.0,
    #         evaluation_data=None,
    #         monitor_evaluation_cost=False,
    #         monitor_evaluation_accuracy=False,
    #         monitor_training_cost=False,
    #         monitor_training_accuracy=False):
    #         ):
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        # 如果有测试数据
        if test_data:
            # 取得测试数据的项数
            n_test = len(test_data)
        # 取得训练数据的项数
        n = len(training_data)

        # in后面是一个迭代器生成器
        # 有多少轮训练，运行多少次
        for j in xrange(epochs):
            # 每一轮操作，打乱训练数据
            random.shuffle(training_data)
            # 分小组，列表生成器每一次得到一个k开始特定长度的小组，生成列表。
            mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            # 小组列表中的每一个小组，都在循环中依次对参数团进行调整，理想中，如果第一个随机集合选的好，那后面的小组准确度可以迅速的提高。
            for mini_batch in mini_batches:
                # 训练小组里面的内容，使用eta
                self.update_mini_batch(mini_batch, eta)
            # 如果没有测试数据，那就结束本轮训练。
            if test_data:
                print "Epoch{0}:{1}/{2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch{0}  complete".format(j)

    # 根据提供的小组数据，计算出一个参数需要变化的值团，eta控制变化幅度的加倍。
    def update_mini_batch(self, mini_batch, eta):
        # def update_mini_batch(self,mini_batch,eta,lmbda,n):
        # 这两句出现突兀，实际上就是初始变量，偏移和权重值
        # 偏移值梯度,获取填充为0的，相同维度的阶数量b组成的列表
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # 权重值梯度,获取填充为0的，相同维度和阶数的量b 组成的列表
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 每一个图片和内容
        for x, y in mini_batch:
            # 返回两个列表，列表元素为每层的偏移量对误差贡献度向量，权重对误差贡献度矩阵。
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 把每次的误差贡献度按元素累加起来
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # 累积起来的值除以数量，得到的就是平均值啦，
        # 降低误差的方法就是，根据自己对总得到的误差的贡献度，减去那部分就好，完成修正。
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    # 使用模型，根据数据团中数据计算出的结果和数据团中标准答案的差别，反推出参数团需要进行的调整。
    def backprop(self, x, y):
        # 初始化一个参数团 (nabla_b,nabla_w)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 活跃值向量，默认为输入向量
        activation = x
        # 活跃值向量列表（每层为一项）
        activations = [x]
        # 存放矩阵变换的输出向量
        zs = []
        # 网络模型进行逐层计算
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            # 使用S型函数压缩
            activation = sigmoid(z)
            # 存储本层的活跃值
            activations.append(activation)
        # 得到两个列表，zs是矩阵变换出活跃值向量列表，activations是压缩的活跃值向量变量列表。
        # 首先delta是权重化输出值的误差贡献度的向量，输出层的神经元数为长度。
        # 前项是误差对输出层活跃值的导数的向量，后一项是由S型函数对权重化输出值的导数的向量
        # 按元素乘完之后，得到输出层误差对权重化输出值的导数向量。
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # 输出层偏移量误差贡献度正是当前层权重化值的误差贡献度
        nabla_b[-1] = delta
        # 转置构成矩阵
        # 输出层的权重误差贡献度，权重化输出的误差贡献度向量提供行，左层活跃值向量提供列，分别乘对应的权重化输出的误差贡献度。
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 从倒数第二层开始
        for l in xrange(2, self.num_layers):
            # 权重化输出向量
            z = zs[-l]
            # S型函数在z上求导
            sp = sigmoid_prime(z)
            # 权重化输出的误差贡献程度是右层权重矩阵转置，右层权重化误差贡献度点乘结果，得到向量 再和S型函数在z上求导值向量，按元素乘。
            # delta j个 当前k个。    jxk->kxj        jx1                     kx1                         k个
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            # 设置当前层的偏移量贡献度
            nabla_b[-l] = delta
            # 设置当前层的权重矩阵误差贡献度：使用左侧输出量和权重化输出的误差贡献度点乘
            # 构成矩阵，点乘转置 kxm         mx1            kx1
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        # 直到输入层设置完毕
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # 返回的是误差对活跃值的导数
    def cost_derivative(self, output_activations, y):
        # 返回误差对活跃值求导
        return (output_activations - y)


# 这是一个定义在全局范围的函数。里面是S型函数的式子。Z是一个当前层神经元的活跃值向量。Numpy的函数会自动应用到每一个元素上，得到的也是向量。
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = neuralnetwork([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


