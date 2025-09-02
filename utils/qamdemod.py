import numpy as np

def generate_gray_code(n_bits):
    """生成 n_bits 位的格雷码列表（字符串形式）"""
    if n_bits == 1:
        return ["0", "1"]
    else:
        prev_gray = generate_gray_code(n_bits - 1)
        return ["0" + code for code in prev_gray] + ["1" + code for code in reversed(prev_gray)]

def qam_demodulation(x_qam_modulated, M):
    """
    QAM 解调
    :param x_qam_modulated: QAM 星座点 (复数数组)
    :param M: 调制阶数 (如 16, 64)
    :return: 解调得到的比特序列 (list[int])
    """
    n_bits = int(np.log2(M))   # 每个符号的比特数
    bits_per_axis = n_bits // 2

    # 生成 Gray code
    gray_code = generate_gray_code(bits_per_axis)

    # 生成星座点的幅度坐标，例如 [-3, -1, +1, +3]（16-QAM）
    AMi_vector = np.array([2 * m - 1 - len(gray_code) for m in range(1, len(gray_code) + 1)])

    bitstream = []

    for qam_element in x_qam_modulated:
        inphase_value = np.real(qam_element)
        quadrature_value = np.imag(qam_element)

        # 找到最近的幅度坐标（I 分量）
        idx_I = np.argmin(np.abs(inphase_value - AMi_vector))
        # 找到最近的幅度坐标（Q 分量）
        idx_Q = np.argmin(np.abs(quadrature_value - AMi_vector))

        # 拼接 Gray code
        sequence = gray_code[idx_I] + gray_code[idx_Q]
        bitstream.extend([int(b) for b in sequence])

    return bitstream
