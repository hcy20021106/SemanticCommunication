# -*- coding: utf-8 -*-
"""
读取一个 Cosmos latent 的 .bin（np.savez 容器，含 packed/shape/bits），
还原为 token 张量，标准化为 [B, L, T, H, W]，
再把每一层按时间步 T 切片保存为 .npy / .csv。

CSV 支持 3 种模式：
- "bits"     ：按二进制位串输出，每个 token 使用 bits_per_symbol 位，前导零齐全
- "decimal"  ：十进制整数
- "hex"      ：十六进制字符串，位宽向上取整到 nibble

!!! 重点：控制 CSV 输出格式的行在下面配置区的 CSV_MODE 变量处（已用醒目标注） 
"""

import os
import numpy as np
import torch

# ============= 需要你修改的配置 ======================
BIN_PATH = "/Huang_group/zyyz/Projects/Wireless-Channel/video_stream/latents_chunk_0_20.bin"  # 你的 .bin(npz) 文件路径
OUT_DIR  = "cut_video_stream"                   # 相对路径，输出目录（会自动创建）
SELECT_LAYERS = None                           # None=导出全部层；或用 [0,2] 只导某些层；或 1 只导第1层

SAVE_NPY = True                                # 是否保存 .npy（建议保留，供程序后续处理）
SAVE_CSV = True                                # 是否保存 .csv

# ========= 这行控制 CSV 输出格式（bits/decimal/hex）=========
CSV_MODE = "bits"   # <<< 控制 CSV 输出：改为 "bits" / "decimal" / "hex"
# ===========================================================

CSV_DELIMITER = ","                            # CSV 分隔符
SAVE_PACKED_BYTES_CSV = True                  # 可选：把打包后的原始字节流另存为按位 CSV（调试/校验用）
PACKED_BYTES_CSV_PATH = "packed_bytes.csv"
# ==========================================================


# ----------------- 工具函数：bit 解包 ----------------------
def unpackbits_torch(packed_bytes: np.ndarray, shape: tuple, bits_per_symbol: int) -> torch.Tensor:
    """将定长 bit-packed 的 bytes 还原成整型张量（CPU）。"""
    total_vals = int(np.prod(shape))
    total_bits = total_vals * bits_per_symbol
    bitarr = np.unpackbits(packed_bytes)[:total_bits]
    vals = np.empty(total_vals, dtype=np.int32)
    bp = bits_per_symbol
    # 将每 bits_per_symbol 位拼回一个整数（高位在前）
    for i in range(0, total_bits, bp):
        v = 0
        for b in range(bp):
            v = (v << 1) | int(bitarr[i + b])
        vals[i // bp] = v
    arr = vals.reshape(shape)
    return torch.from_numpy(arr)

def load_tokens_npz(bin_path: str):
    """
    从 .bin(npz) 载入 packed/shape/bits，并解包为整型 torch.Tensor（CPU）。
    返回: tokens, bits_per_symbol, packed_bytes
    """
    with np.load(bin_path, allow_pickle=True) as z:
        packed = z["packed"]
        shape  = tuple(int(x) for x in z["shape"])
        bits   = int(z["bits"])
    tokens = unpackbits_torch(packed, shape, bits)
    return tokens, bits, packed


# -------- 形状标准化到 [B, L, T, H, W] + 选择层 --------
def standardize_B_L_T_H_W(tokens: torch.Tensor) -> torch.Tensor:
    """
    将各种常见形状转为 [B, L, T, H, W]：
      - [B, L, T, H, W] 直接返回
      - [B, T, H, W] -> 视为 L=1，在 dim=1 插一维
      - [B, T, L, H, W] / [B, L, H, W, T] 等进行启发式重排
    """
    dims = list(tokens.shape)
    if len(dims) == 5:
        B, d1, d2, d3, d4 = dims
        # 经验：L 通常较小（<=16），T 通常是较大的维
        if d1 <= 16 and d2 >= d3 and d2 >= d4:
            return tokens  # [B, L, T, H, W]
        if d2 <= 16 and d1 > d2:
            return tokens.permute(0, 2, 1, 3, 4)  # [B, L, T, H, W]
        if d1 <= 16 and d4 >= d2 and d4 >= d3:
            return tokens.permute(0, 1, 4, 2, 3)  # [B, L, T, H, W]
        return tokens
    elif len(dims) == 4:
        return tokens.unsqueeze(1)  # [B, 1, T, H, W]
    else:
        raise ValueError(f"Unexpected token shape: {tokens.shape}")

def normalize_select_layers(select):
    """把 SELECT_LAYERS 规范成列表或 None。"""
    if select is None:
        return None
    if isinstance(select, int):
        return [select]
    if isinstance(select, (list, tuple)):
        return list(select)
    raise ValueError("SELECT_LAYERS 应为 None / int / list[int]")


# ---------- CSV 辅助：把整数矩阵转成字符串矩阵 ----------
def _ints_to_bitstrings_2d(arr: np.ndarray, width: int) -> np.ndarray:
    """二维整型矩阵 -> 等宽二进制字符串矩阵（零填充到 width 位）。"""
    flat = arr.reshape(-1)
    out = np.array([format(int(x), f"0{width}b") for x in flat], dtype=f"<U{width}")
    return out.reshape(arr.shape)

def _ints_to_hexstrings_2d(arr: np.ndarray, width_bits: int) -> np.ndarray:
    """二维整型矩阵 -> 等宽十六进制字符串矩阵（宽度=ceil(width_bits/4)）。"""
    w_hex = (width_bits + 3) // 4  # 每4位1个hex
    flat = arr.reshape(-1)
    out = np.array([format(int(x), f"0{w_hex}X") for x in flat], dtype=f"<U{w_hex}")
    return out.reshape(arr.shape)


# --------------- 保存按 T 切片（NPY / CSV） ---------------
def save_T_slices(tokens_BLTHW: torch.Tensor, bits_per_symbol: int,
                  out_dir: str, select_layers=None):
    """
    保存为 out_dir/layer{l}/T_{t:05d}.(npy/csv)
    - CSV_MODE="bits"    ：每个 token 以二进制位串写出，宽度=bits_per_symbol
    - CSV_MODE="decimal" ：十进制
    - CSV_MODE="hex"     ：十六进制、等宽
    """
    B, L, T, H, W = tokens_BLTHW.shape
    assert B == 1, f"当前脚本假设 B=1，实际 B={B}"
    os.makedirs(out_dir, exist_ok=True)

    sel = normalize_select_layers(select_layers)
    layer_indices = sel if sel is not None else list(range(L))

    print(f"[信息] 形状：B={B}, L={L}, T={T}, H={H}, W={W}, bits={bits_per_symbol}")
    print(f"[信息] 导出层：{layer_indices}，CSV_MODE={CSV_MODE}（由配置区那一行控制）")

    for l in layer_indices:
        layer_dir = os.path.join(out_dir, f"layer{l}")
        os.makedirs(layer_dir, exist_ok=True)
        grid_THW = tokens_BLTHW[0, l].numpy()  # [T, H, W] int32

        for t in range(T):
            arr = grid_THW[t]  # [H, W]

            # 保存 NPY
            if SAVE_NPY:
                np.save(os.path.join(layer_dir, f"T_{t:05d}.npy"), arr)

            # 保存 CSV（根据 CSV_MODE 决定格式）
            if SAVE_CSV:
                csv_path = os.path.join(layer_dir, f"T_{t:05d}.csv")
                if CSV_MODE == "bits":
                    s = _ints_to_bitstrings_2d(arr, width=bits_per_symbol)
                    np.savetxt(csv_path, s, fmt="%s", delimiter=CSV_DELIMITER)
                elif CSV_MODE == "hex":
                    s = _ints_to_hexstrings_2d(arr, width_bits=bits_per_symbol)
                    np.savetxt(csv_path, s, fmt="%s", delimiter=CSV_DELIMITER)
                elif CSV_MODE == "decimal":
                    np.savetxt(csv_path, arr, fmt="%d", delimiter=CSV_DELIMITER)
                else:
                    raise ValueError(f"未知 CSV_MODE: {CSV_MODE}")

        print(f"[完成] layer {l}: 保存 {T} 个切片 -> {layer_dir}")


# ------------------------------ 主流程 ------------------------------
if __name__ == "__main__":
    # 1) 读取并解包（拿到 bits_per_symbol 与 packed）
    tokens, bits_per_symbol, packed = load_tokens_npz(BIN_PATH)

    # （可选）把打包后的原始字节流也导出一份二进制位串 CSV（便于核对）
    if SAVE_PACKED_BYTES_CSV:
        bytes_arr = np.frombuffer(packed, dtype=np.uint8)
        # 每行 16 字节更易读；不需要可省略折行
        ncols = 16
        pad = (-len(bytes_arr)) % ncols
        if pad:
            bytes_arr = np.pad(bytes_arr, (0, pad), constant_values=0)
        bytes_mat = bytes_arr.reshape(-1, ncols)
        # 将每字节格式化为 8 位二进制字符串
        bytes_str = np.vectorize(lambda x: format(int(x), "08b"))(bytes_mat)
        np.savetxt(PACKED_BYTES_CSV_PATH, bytes_str, fmt="%s", delimiter=CSV_DELIMITER)
        print(f"[信息] 已导出 packed 原始字节 -> {PACKED_BYTES_CSV_PATH}")

    # 2) 标准化形状到 [B, L, T, H, W]
    tokens = standardize_B_L_T_H_W(tokens)

    # 3) 按 T 切片保存（CSV 输出格式由 CSV_MODE 那一行控制）
    save_T_slices(tokens, bits_per_symbol, OUT_DIR, select_layers=SELECT_LAYERS)

    print("[OK] Done.")
