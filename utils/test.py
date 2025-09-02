from qammod import QAM_modulation
from qamdemod import qam_demodulation
# 示例：16-QAM
bits = [0,0, 1,1, 0,1, 1,0]  # 共8比特
symbols = QAM_modulation(bits, 16)
print(symbols)
bits = qam_demodulation(symbols, 16)
print(bits)