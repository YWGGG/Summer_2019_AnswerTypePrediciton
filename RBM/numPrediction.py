import matplotlib.pylab as plt
import numpy as np
import struct
import random

class RBM(object):
    def __init__(self,n_visible,n_hidden,momentum = 0.5,learning_rate = 0.1,max_epoch = 50,batch_size = 128,penalty = 0,weight = None,
                bias_visible = None,bias_hidden = None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.momentum = momentum

        if weight is None:
            self.weight = np.random.random((self.n_hidden,self.n_visible))*0.1
        else:
            self.weight =weight
        if bias_visible is None:
            self.bias_visible = np.zeros(self.n_visible)
        else:
            self.bias_visible = bias_visible
        if bias_hidden is None:
            self.bias_hidden = np.zeros(self.n_hidden)
        else:
            self.bias_hidden = bias_hidden

    def sigmoid(self , z):
        return 1.0/(1.0+np.exp(-z))


    def forward(self , x):
        z = np.dot(x,self.weight.T)+ self.bias_hidden
        return self.sigmoid(z)

    def backward(self, y):
        z = np.dot(y,self.weight) +self.bias_visible
        return self.sigmoid(z)




    def batch(self):
        m,n = self.input_x.shape

        index = list(range(m))
        random.shuffle(index)
        index = [index[k:k+self.batch_size] for k in range(0,m,self.batch_size)]

        batch_data = []
        for group in index:
            batch_data.append(self.input_x[group])
        return batch_data

    def fit(self, input_x):
        self.input_x = input_x

        Winc = np.zeros_like(self.weight)
        binc = np.zeros_like(self.bias_visible)
        cinc = np.zeros_like(self.bias_hidden)

        for epoch in range(self.max_epoch):
            batch_data = self.batch()
            num_batches = len(batch_data)

            err_sum = 0.0
            self.penalty = (1-0.9*epoch/self.max_epoch)*self.penalty

            for v0 in batch_data:
                # forward
                h0 = self.forward(v0)
                h0_states = np.zeros_like(h0)

                h0_states[h0 > np.random.random(h0.shape)] = 1

                #backforward
                v1 = self.backward(h0_states)
                v1_states = np.zeros_like(v1)
                v1_states[v1 > np.random.random(v1.shape)] = 1

                #forward
                h1 = self.forward(v1_states)
                h1_states = np.zeros_like(h1)
                h1_states[h1 > np.random.random(h1.shape)] = 1

                #update
                dW = np.dot(h0_states.T, v0) - np.dot(h1_states.T, v1)

                db = np.sum(v0 - v1, axis=0).T
                dc = np.sum(h0 - h1, axis=0).T

                #速度更新
                Winc = self.momentum * Winc + self.learning_rate * (dW - self.penalty * self.weight) / self.batch_size
                binc = self.momentum * binc + self.learning_rate * db / self.batch_size
                cinc = self.momentum * cinc + self.learning_rate * dc / self.batch_size

                self.weight = self.weight + Winc
                self.bias_visible = self.bias_visible + binc
                self.bias_hidden = self.bias_hidden + cinc

                err_sum = err_sum + np.mean(np.sum((v0 - v1) ** 2, axis=1))

            #平均误差
            err_sum = err_sum / num_batches
            print('Epoch {0},err_sum {1}'.format(epoch, err_sum))



    def predict(self,input_x):
        h0 = self.forward(input_x)
        h0_states = np.zeros_like(h0)
        h0_states[h0 > np.random.random(h0.shape)] = 1

        #反向
        v1 = self.backward(h0_states)
        return v1

    def visualize(self, input_x):
        m, n = input_x.shape
        s = int(np.sqrt(n))

        row = int(np.ceil(np.sqrt(m)))

        data = np.zeros((row * s + row + 1, row * s + row + 1)) - 1.0

        # 图像在x轴索引
        x = 0
        # 图像在y轴索引
        y = 0
        # 遍历每一张图像
        for i in range(m):
            z = input_x[i]
            z = np.reshape(z, (s, s))

            data[x * s + x + 1:(x + 1) * s + x + 1, y * s + y + 1:(y + 1) * s + y + 1] = z
            x = x + 1
            # 换行
            if (x >= row):
                x = 0
                y = y + 1
        return data





def fetch_mnist(mnist_dir,data_type):
    train_data_path = mnist_dir + 'train-images.idx3-ubyte'
    train_label_path = mnist_dir + 'train-labels.idx1-ubyte'
    test_data_path = mnist_dir + 't10k-images.idx3-ubyte'
    test_label_path = mnist_dir + 't10k-labels.idx1-ubyte'

    # train_img
    with open(train_data_path, 'rb') as f:
        data = f.read(16)
        des,img_nums,row,col = struct.unpack_from('>IIII', data, 0)
        train_x = np.zeros((img_nums, row*col))
        for index in range(img_nums):
            data = f.read(784)
            if len(data) == 784:
                train_x[index,:] = np.array(struct.unpack_from('>' + 'B' * (row * col), data, 0)).reshape(1,784)
        f.close()
    # train label
    with open(train_label_path, 'rb') as f:
        data = f.read(8)
        des,label_nums = struct.unpack_from('>II', data, 0)
        train_y = np.zeros((label_nums, 1))
        for index in range(label_nums):
            data = f.read(1)
            train_y[index,:] = np.array(struct.unpack_from('>B', data, 0)).reshape(1,1)
        f.close()

        # test_img
        with open(test_data_path, 'rb') as f:
            data = f.read(16)
            des, img_nums, row, col = struct.unpack_from('>IIII', data, 0)
            test_x = np.zeros((img_nums, row * col))
            for index in range(img_nums):
                data = f.read(784)
                if len(data) == 784:
                    test_x[index, :] = np.array(struct.unpack_from('>' + 'B' * (row * col), data, 0)).reshape(1, 784)
            f.close()
        # test label
        with open(test_label_path, 'rb') as f:
            data = f.read(8)
            des, label_nums = struct.unpack_from('>II', data, 0)
            test_y = np.zeros((label_nums, 1))
            for index in range(label_nums):
                data = f.read(1)
                test_y[index, :] = np.array(struct.unpack_from('>B', data, 0)).reshape(1, 1)
            f.close()
        if data_type == 'train':
            return train_x, train_y
        elif data_type == 'test':
            return test_x, test_y
        elif data_type == 'all':
            return train_x, train_y,test_x, test_y
        else:
            print('type error')




if __name__ == '__main__':
    # 加载MNIST数据集
    tr_x, tr_y, te_x, te_y = fetch_mnist('', 'all')


    data = np.array(tr_x)
    print(data.shape)  # (60000, 784)

    # 创建RBM网络
    rbm = RBM(784, 100, max_epoch=50, learning_rate=0.05)
    # 开始训练
    rbm.fit(data)

    # 显示64张手写数字
    images = data[0:64]
    print(images.shape)
    a = rbm.visualize(images)
    fig = plt.figure(1, figsize=(8, 8))
    plt.imshow(a, cmap=plt.cm.gray)
    plt.title('original data')

    # 显示重构的图像
    rebuild_value = rbm.predict(images)
    b = rbm.visualize(rebuild_value)
    fig = plt.figure(2, figsize=(8, 8))
    plt.imshow(b, cmap=plt.cm.gray)
    plt.title('rebuild data')

    # 显示权重
    w_value = rbm.weight
    c = rbm.visualize(w_value)
    fig = plt.figure(3, figsize=(8, 8))
    plt.imshow(c, cmap=plt.cm.gray)
    plt.title('weight value(w)')
    plt.show()