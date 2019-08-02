import numpy as np
import struct
mnist_dir = ''
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
    tr_x, tr_y, te_x, te_y = fetch_mnist(mnist_dir,'all')
    import matplotlib.pyplot as plt # plt 用于显示图片
    img_0 = tr_x[59999,:].reshape(28,28)
    plt.imshow(img_0)
    print(tr_y[59999,:])
    img_1 = te_x[500,:].reshape(28,28)
    plt.imshow(img_1)
    print(te_y[500,:])
    plt.show()