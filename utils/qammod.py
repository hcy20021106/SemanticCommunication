import numpy as np

def generate_gray_code(n_bits):
    """生成 n_bits 位的格雷码表，返回字符串列表"""
    if n_bits == 0:
        return ['']
    if n_bits == 1:
        return ['0', '1']
    prev_gray = generate_gray_code(n_bits - 1)
    return ['0' + code for code in prev_gray] + ['1' + code for code in reversed(prev_gray)]

def QAM_modulation(x_encoded, M):
    """
    QAM 调制函数
    :param x_encoded: 输入比特序列 (list 或 numpy array, 元素为 0/1)
    :param M: QAM 阶数 (如 16 表示 16-QAM)
    :return: 复数调制信号数组
    """
    x_encoded = list(map(str, x_encoded))  # 转成字符串方便拼接
    bits_per_symbol = int(np.log2(M))      # 每个符号的比特数
    
    # I/Q 各自占一半比特
    n_bits_axis = bits_per_symbol // 2
    greycode = generate_gray_code(n_bits_axis)

    # 填充使比特数是符号比特的整数倍
    if len(x_encoded) % bits_per_symbol != 0:
        pad_len = bits_per_symbol - (len(x_encoded) % bits_per_symbol)
        x_encoded.extend(['0'] * pad_len)

    n_symbols = len(x_encoded) // bits_per_symbol
    x_QAM_modulated = np.zeros(n_symbols, dtype=complex)

    # 构造 PAM 电平向量，例如 [-3, -1, +1, +3]
    levels = np.array([2*m - 1 - len(greycode) for m in range(1, len(greycode)+1)])

    k = 0
    for i in range(n_symbols):
        seq = x_encoded[k:k+bits_per_symbol]
        k += bits_per_symbol

        # 前一半比特 → I，后一半比特 → Q
        seq_even = ''.join(seq[:n_bits_axis])
        seq_odd  = ''.join(seq[n_bits_axis:])

        idx_even = greycode.index(seq_even)
        idx_odd  = greycode.index(seq_odd)

        AMi = levels[idx_even]
        AMq = levels[idx_odd]

        x_QAM_modulated[i] = AMi + 1j * AMq

    return x_QAM_modulated
