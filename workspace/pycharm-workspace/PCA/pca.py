import numpy as np


def feature_Normalize(x):
    """
    归一化数据
    (每个数据-当前列的均值)/当前列的标准差
    :param x: 样本集
    :return: 归一化后样本集,均值,标准差
    """
    m, n = x.shape
    mean = np.zeros((1, n))
    std = np.zeros((1, n))
    # 计算各列均值
    mean = np.mean(x, axis=0)
    # 计算各列标准差
    std = np.std(x, axis=0)
    # 对每个特征值归一化
    for i in range(n):
            x[:, i] = (x[:, i] - mean[i]) / std[i]
    return x, mean, std


def cal_eigenvalue(nor_x):
    """
    求样本协方差矩阵的特征值和特征向量
    :param nor_x: 归一化后的样本集
    :return: 特征值,特征向量,排序索引号
    """
    m, n = nor_x.shape
    # 协方差矩阵
    sigma = np.dot(np.transpose(nor_x), nor_x)/(m - 1)
    # 求协方差矩阵的特征值和特征向量,eig_vec[:,i]是对应于eig_val[i]的特征向量
    eig_val, eig_vec = np.linalg.eig(sigma)
    index = eig_val.argsort()
    return eig_val, eig_vec, index


def pca(x, k):
    """
    提取前k个主成分
    :param x: 样本集
    :param k: 前k个特征值
    :return: 返回降维后样本,累计贡献度,主成分索引
    """
    # 归一化
    nor_x, mean, std = feature_Normalize(x)
    # 求特征值和特征向量
    eig_val, eig_vec, index = cal_eigenvalue(nor_x)
    eig_index = index[:-(k+1):-1]
    # 累计贡献度
    sum_con = sum(eig_val[eig_index])/sum(eig_val)
    # 前k个特征值对应的特征向量
    k_eig_vec = eig_vec[:, eig_index]
    lowDData = np.dot(nor_x, k_eig_vec)
    return lowDData, sum_con, eig_index
