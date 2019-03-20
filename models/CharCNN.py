# @Author : tw19941212
# @Datetime : 2019/01/13 20:37
# @Last Modify Time : 2019/01/13 20:37
# @paper : Character-level Convolutional Networks for Text Classification
# @paper link : https://arxiv.org/abs/1509.01626
# @blog llink : https://tw19941212.github.io/posts/f07060b1/#more

from keras.models import Model
from keras.layers import Embedding, Dense,  Conv1D, MaxPool1D, ThresholdedReLU
from keras.layers import Input, Dropout, Flatten

# parameter:
conv_layers = [
    [256, 7, 3],
    [256, 7, 3],
    [256, 3, None],
    [256, 3, None],
    [256, 3, None],
    [256, 3, 3]
]  # 卷积层参数,[fliters, kernel_size, pool_size]
fully_layers = [1024, 1024]  # 全连接层参数
alphabet_size = 69  # unique字符总数量
imput_size = 1024  # 最大的输入text的字符长度,不够则pad超长截断
embed_size = 16  # 字符表示序列最大长度
threshold = 0  # ThresholdedReLU 的阈值
num_class = 4  # 输出类别


def get_CharCNN():
    # 输入层
    inp = Input(shape=(imput_size,))
    # Embedding层: alphabet_size+1: 多出来的一个给空字符或非字符表字符
    x = Embedding(alphabet_size+1, embed_size, trainable=True)(inp)
    # 卷积层
    for c in conv_layers:
        x = Conv1D(c1[0], kernel_size=c1[1])(x)
        x = ThresholdedReLU(threshold)(x)
        if c[2] is not None:
            x = MaxPool1D(c1[2])(x)
    x = Flatten()(x)
    # 全连接层
    for f in fully_layers:
        x = Dense(f)(x)
        x = ThresholdedReLU(threshold)(x)
        x = Dropout(0.5)(x)  # 可选择不用正则化
    # 输出层
    outp = Dense(num_class, activation="softmax")(x)
    # 构建,编译 model
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

# CharCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.fully connected layers 5.softmax layer.
