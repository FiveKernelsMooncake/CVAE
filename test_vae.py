import numpy as np
import tensorflow as tf
import scipy.io
import logging
from lib.vae import VariationalAutoEncoder
from lib.utils import *
import json
np.random.seed(0)
tf.set_random_seed(0)
init_logging("vae.log")

logging.info('loading data')
#读取item_contentx信息
#variables = scipy.io.loadmat("data/citeulike-a/mult_nor.mat")
#data 一行代表一篇文章的词袋模型的频率值
#data = variables['X']

data=np.zeros([139176,7983])
#with open(r'D:\360MoveData\Users\刘贝\Desktop\CVPR2020\Data\final_asin_vector_203_5_sorted.json','r') as load_f:
with open("my_data/final_asin_vector_203_5_sorted.json", 'r') as load_f:
        load_dict = json.load(load_f)
#将数据存入data中
count = 0

for i in load_dict:
    for j in i['vector']:
        data[count, j[0]] = j[1]
    count = count + 1

print('data.shape',data.shape)#(16980, 8000)
idx = np.random.rand(data.shape[0]) < 0.8
train_X = data[idx]
print('train_X.shape',train_X.shape)#(13630, 8000)
#一共16980篇文章，分为train：13630 和test：3350
#print(len(train_X))  13630
test_X = data[~idx]
print('test_X.shape',test_X.shape)#(3350, 8000)
#print(len(test_X))  3350
logging.info('initializing sdae model')
model = VariationalAutoEncoder(input_dim=7983, dims=[200, 100], z_dim=50,
	activations=['sigmoid','sigmoid'], epoch=[50, 50],
	noise='mask-0.3' ,loss='cross-entropy', lr=0.01, batch_size=128, print_step=1)
logging.info('fitting data starts...')
model.fit(train_X, test_X)
# feat = model.transform(data)
# scipy.io.savemat('feat-dae.mat',{'feat': feat})
# np.savez("sdae-weights.npz", en_weights=model.weights, en_biases=model.biases,
# 	de_weights=model.de_weights, de_biases=model.de_biases)
