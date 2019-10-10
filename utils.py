import numpy as np
import logging


def init_logging(log_path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)

#将X变为一个地址就可以了，需要多少读进来，再做变换，好像顺序不是很重要，我只要把权重训练明白就行了
#直接根据指定的指来找是不太可能了，改成随机一个开始位置，和它后面的128个吧
#试试看能不能收敛
#X:json文件
def get_batch(X, size):
    ids = np.random.choice(len(X), size, replace=False)
    return (X[ids], ids)
'''
def get_batch(X, size):
    ids = np.random.choice(len(X), 128, replace=False)
    data_u = np.zeros([128, 7983])
    count = 0
    # 读入选取的数据
    XX = []
    for iids in ids:
        XX.append(X[iids])
    for i in XX:
        for j in i['vector']:
            data_u[count, j[0]] = j[1]
        count = count + 1
    return (data_u, ids)
'''

def noise_validator(noise, allowed_noises):
    '''Validates the noise provided'''
    try:
        if noise in allowed_noises:
            return True
        elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
            t = float(noise.split('-')[1])
            if t >= 0.0 and t <= 1.0:
                return True
            else:
                return False
    except:
        return False
    pass
