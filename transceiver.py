import numpy as np
from channel.channel import Channel
from utils.qammod import QAM_modulation
from utils.qamdemod import qam_demodulation

# 1. 读取比特流
with open('T_vectors/slice_000000.txt', 'r') as f:
    bitstream = [int(b) for b in f.read().strip()]

# 2. QAM调制
M = 4
symbols = QAM_modulation(bitstream, M)

# 3. 信道仿真
snr_db = 1
channel = Channel(chan_type='awgn', snr_db=snr_db)
rx_symbols = channel(symbols)

# 4. QAM解调
rx_bits = qam_demodulation(rx_symbols, M)

# 5. 计算BER
bitstream = bitstream[:len(rx_bits)]  # 截断到解调长度
ber = np.mean(np.array(bitstream) != np.array(rx_bits))
print(f'BER: {ber:.6f}')