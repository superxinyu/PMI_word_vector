import numpy as np
from co_occurrence_matrix import get_matrix
from utils import load_data


def get_pmi(file_path, positive=True, return_word_vec=False):
    """
    获得pmi矩阵或者pmi词向量
    :param file_path: 数据路径地址，每行一个句子
    :param positive: 是否要ppmi
    :param return_word_vec: 返回词向量（一个dict）还是pmi矩阵
    :return:
    """
    # 读取数据
    data = load_data(file_path)
    # 得到共现矩阵
    text_matrix, id2word = get_matrix(data, return_word_id=True)
    # 计算pmi矩阵
    column_sum = text_matrix.sum(axis=0)
    row_sum = text_matrix.sum(axis=1)
    all_sum = column_sum.sum()
    # 理解这个式子的关键就是要理解矩阵表示词频，矩阵每一个元素的计算其实都是词频两两间的计算
    # all_sum就是总词频，np.outer(row_sum, column_sum)就是该词与其他词的共现次数
    expected = np.outer(row_sum, column_sum) / all_sum
    text_matrix = text_matrix / expected
    # 忽略log0警告
    with np.errstate(divide='ignore'):
        text_matrix = np.log(text_matrix)
    text_matrix[np.isinf(text_matrix)] = 0.0
    # 是否要ppmi矩阵
    if positive:
        text_matrix[text_matrix < 0] = 0.0
    if return_word_vec:
        word_vec={}
        for i,vec in enumerate(text_matrix):
            word_vec[id2word[i]]=vec
        return word_vec
    return text_matrix
