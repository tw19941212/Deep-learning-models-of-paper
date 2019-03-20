# @Author : tw19941212
# @Datetime : 2019/01/12 16:37
# @Last Modify Time : 2019/01/12 16:37
# @paper : Convolutional Neural Networks for Sentence Classification
# @paper link : https://arxiv.org/abs/1408.5882
# @blog llink : https://tw19941212.github.io/posts/f07060b1/#more

from keras.models import Model
from keras.layers import Embedding, Dense,  Conv1D, GlobalMaxPooling1D
from keras.layers import Input, Dropout, Concatenate, Flatten, SpatialDropout1D

# parameter:
filter_size = [3, 4, 5]  # filterwindows (h) of 3, 4, 5
num_fliters = 100  # 100 feature maps each
max_features = 50  # 最大unique单词数量
embed_size = 300  # 词向量的长度
embedding_matrix = np.zeros(
    (max_features, embed_size))  # 初始化词向量矩阵,实际中是加载预训练词向量
maxlen = 60  # 文本的最大长度


def get_TextCNN():
    # 输入层
    inp = Input(shape=(maxlen,))
    # Embedding层
    x = Embedding(max_features, embed_size, weights=[
                  embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)  # 随机丢弃某些词,起到数据增强作用
    out = []
    # 卷积层
    for size in filter_size:
        CONV = Conv1D(num_fliters, kernel_size=size,
                      kernel_initializer='he_normal', activation='tanh')(x)
        out.append(GlobalMaxPooling1D()(CONV))
    x = Concatenate(axis=1)(out)
    x = Dropout(0.5)(x)  # 可选择不用正则化
    # 输出层
    outp = Dense(1, activation="sigmoid")(x)
    # 构建,编译 model
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

# TextCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.sigmoid layer.
# 文本分类任务在卷积层前加RNN(LSTM,GRU)能取得更好的效果
