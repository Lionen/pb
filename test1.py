# coding=utf-8
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import keras  # 显示图中的节点

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train, label = mnist.train.next_batch(20000)

model = Sequential()
model.add(Dense(256, name="x", input_shape=(784,)))
model.add(Dense(10, activation="softmax", name="y"))

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.fit(train, label, batch_size=64, epochs=1)



print(model.input.name)
print(model.output.name)

sess = K.get_session()

frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,
    sess.graph_def,
    output_node_names=[ "y/Softmax"])

# 保存图为pb文件
with open('model/model.pb', 'wb') as f:
    f.write(frozen_graph_def.SerializeToString())
