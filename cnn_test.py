'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
# 使用Sequential模型
from keras.models import Sequential
# 导入Dense，Dropout，Flatten，Conv2D，MaxPooling2D层
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# 调用后端接口
from keras import backend as K
import tensorflow as tf

# batch大小，每处理128个样本进行一次梯度更新
batch_size = 128
# 类别数
num_classes = 10
# 迭代次数
epochs = 1

# input image dimensions
# 28x28 图像
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# -x_train.shape:(6000, 28, 28)
# -x_test.shape:(1000, 28, 28)

# tf或th为后端，采取不同参数顺序
if K.image_data_format() == 'channels_first':
    # -x_train.shape[0]=6000
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    # -x_train.shape:(60000, 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    # x_test.shape:(10000, 1, 28, 28)
    # 单通道灰度图像,channel=1
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 数据转为float32型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 归一化
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# 标签转换为独热码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 构建模型
model = Sequential()
# 第一层为二维卷积层
# 32 为filters卷积核的数目，也为输出的维度
# kernel_size 卷积核的大小，3x3
# 激活函数选为relu
# 第一层必须包含输入数据规模input_shape这一参数，后续层不必包含
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 name='x',
                 input_shape=input_shape))
# 再加一层卷积，64个卷积核
model.add(Conv2D(64, (3, 3), activation='relu'))
# 加最大值池化
model.add(MaxPooling2D(pool_size=(2, 2)))
# 加Dropout，断开神经元比例为25%
model.add(Dropout(0.25))
# 加Flatten，数据一维化
model.add(Flatten())
# 加Dense，输出128维
model.add(Dense(128, activation='relu'))
# 再一次Dropout
model.add(Dropout(0.5))
# 最后一层为Softmax，输出为10个分类的概率
model.add(Dense(num_classes, activation='softmax', name='y'))

# 配置模型，损失函数采用交叉熵，优化采用Adadelta，将识别准确率作为模型评估
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 训练模型，载入数据，verbose=1为输出进度条记录
# validation_data为验证集
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# 开始评估模型效果
# verbose=0为不输出日志信息
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
# model.save('model/cnn_model.h5')
#

print(model.predict(x_test[0].reshape(1, 28, 28, 1)))
print(model.input.name)
print(model.output.name)

sess = K.get_session()

frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,
    sess.graph_def,
    output_node_names=["y/Softmax"])

# frozen_graph_def = tf.get_default_graph().as_graph_def()

# var_list = tf.global_variables()
# # for i in var_list:
# #     print(i)
# frozen_graph_def = tf.graph_util.convert_variables_to_constants(
#     sess, sess.graph_def, output_node_names=[var_list[i].name for i in range(len(var_list))])

# 保存图为pb文件
# with open('model/cnn_model.pb', 'wb') as f:
#     f.write(frozen_graph_def.SerializeToString())

tf.train.write_graph(frozen_graph_def, 'model', 'cnn_test_model.pb', as_text=False)
