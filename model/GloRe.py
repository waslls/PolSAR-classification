import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class GloRe(object):
    def __init__(self, x=None, reduced_dim=None, num_node=None):
        super(GloRe, self).__init__()
        self.mapping_to_B = layers.Conv2D(num_node, kernel_size=1)
        self.reduce_dim = layers.Conv2D(reduced_dim, kernel_size=1)
        self.extened_dim = layers.Conv2D(x.shape[3], kernel_size=1)
        self.gcn = GCN(2*reduced_dim, num_node)
        self.num_n = num_node
        self.num_s = reduced_dim

    def forward(self, x):
        #x.shape:(B,H,W,C) B.shape:(B,N,H,W)  N:num_node C':reduced_dim图结构初始维度
        B = tf.transpose(self.mapping_to_B(x), [0, 3, 1, 2])
        #reduced_x.shape:(B,H,W,C')
        reduced_x = self.reduce_dim(x)
        #B_reshaped:(B,N,H*W)   reduced_x_reshaped:(B,H*W,C')
        n = x.shape[0]
        B_reshaped = tf.reshape(B, [-1, self.num_n, x.shape[1]*x.shape[2]])
        reduced_x_reshaped = tf.reshape(reduced_x, [-1, x.shape[1]*x.shape[2], self.num_s])
        #v.shape:(B,N,C')
        v = tf.matmul(B_reshaped, reduced_x_reshaped)
        #v_transformed.shape:(B,N,2*C')
        v_transformed = self.gcn.forward(v)
        #Y.shape:(B,H*W,2*C')
        Y = tf.matmul(tf.transpose(B_reshaped, [0, 2, 1]), v_transformed)
        #Y_reshaped.shape:(B,H,W,2*C')
        Y_reshaped = tf.reshape(Y, [-1, x.shape[1], x.shape[2], 2*self.num_s])
        #Y_extend_dim.shape:(B,H,W,x.shape[3])
        Y_extend_dim = self.extened_dim(Y_reshaped)
        return x+Y_extend_dim



class GCN(object):
    '''
    num_state:为图卷积后的维度
    '''
    def __init__(self, num_state=None, num_node=None, bias=False):
        super(GCN, self).__init__()
        self.conv1 = layers.Conv1D(num_node, kernel_size=1)
        self.conv2 = layers.Conv1D(num_state, kernel_size=1, use_bias=bias, activation='relu')

    def forward(self, x):
        #x.shape:B ,N ,C'
        h = self.conv1(tf.transpose(x, [0, 2, 1]))#先对节点维度作1*1卷积 哪个维度没变 就是对哪个维度作1*1卷积
        #h.shape:B ,C',N
        h = tf.transpose(h, [0, 2, 1])
        #h.shape:B ,N ,C'
        h = h-x#经验发现效果更好
        #h.shape:B ,N ,2*C'
        h = self.conv2(h)
        return h

def main():
    a = np.ones((8, 32, 32, 1024)).astype(np.float32)#tf输入需要32
    m = GloRe(a, reduced_dim=512, num_node=80)
    print(m.forward(a).shape)

if __name__=="__main__":
    main()
