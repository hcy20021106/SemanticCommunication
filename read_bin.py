#-----------------------------文件说明------------------------------
# 把bin 转换成 T个npy一维的文件，再把T个npy转换成bin码流
# ------------------------------------------------------------------
# bin_npy_roundtrip.py
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import struct, io

# ===== 你可以在这里改路径 =====
BIN_INPUT_PATH  = "video_stream/latents_chunk_0_20.bin"      # bin -> 多个 .npy
NPY_OUTPUT_DIR  = "T_vectors"                                 # 切片输出目录

NPY_INPUT_DIR   = "T_vectors"                                 # 多个 .npy -> bin
BIN_OUTPUT_PATH = "video_stream/restored_latents_chunk_0_20.bin"  # 还原后的 .bin（确保就是 .bin）

# ===== 固定参数（与写切片时保持一致）=====
BITORDER = "big"     # 比特端序
HDR_BYTES = 5        # 头 5 字节: bps, T, t, H, W
HDR_BITS  = HDR_BYTES * 8

def split_bin_to_npy(bin_path, out_dir):
    """
    读取 .bin(=NPZ) -> 按时间维 T 切 T 段 -> 每段存成一维 bit 向量(.npy)
    头部：>BBBBB (bps, T, t, H, W)，共 40 bits；载荷：该片 H*W*bps 个比特。
    """
    bin_path = Path(bin_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(bin_path, allow_pickle=True) as f:
        packed = f["packed"].astype(np.uint8, copy=False)
        shape  = tuple(int(x) for x in np.array(f["shape"]).ravel())
        bps    = int(f["bits"])

    if len(shape) == 4:
        B, T, H, W = shape
    elif len(shape) == 3:
        T, H, W = shape; B = 1
    else:
        raise ValueError(f"不支持的 latent 形状: {shape}")
    assert B == 1, "本脚本假设 B=1"

    total_symbols = B * T * H * W
    total_bits = total_symbols * bps
    bits_all = np.unpackbits(packed, bitorder=BITORDER)[:total_bits]
    bits_per_slice = H * W * bps

    if any(x > 255 for x in (bps, T, H, W)):
        raise ValueError(f"头字段要求 <=255，但得到 bps={bps}, T={T}, H={H}, W={W}")

    for t in range(T):
        start = t * bits_per_slice
        end   = start + bits_per_slice
        slice_bits = bits_all[start:end]  # (H*W*bps,)

        header = struct.pack(">BBBBB", bps, T, t, H, W)
        head_bits = np.unpackbits(np.frombuffer(header, dtype=np.uint8), bitorder=BITORDER)

        vec_bits = np.concatenate([head_bits, slice_bits]).astype(np.uint8)  # 一维0/1
        np.save(out_dir / f"slice_{t:06d}.npy", vec_bits)

        if t < 3 or t == T-1:
            print(f"[split] slice {t:6d}: len={vec_bits.size} bits "
                  f"(头={HDR_BITS}, 载荷={slice_bits.size}) -> slice_{t:06d}.npy")

    print(f"[OK] 已输出 {T} 个 .npy 到 {out_dir.resolve()}")
    return {"T": T, "H": H, "W": W, "bps": bps, "out_dir": str(out_dir.resolve())}

def join_npy_to_bin(npy_dir, out_bin_path):
    """
    从目录读取所有 .npy 切片（每个是 头40bits + 载荷bits），
    按 t 排序拼接，打包回 packed，并把“NPZ字节”写入一个真正的 .bin 文件。
    """
    npy_dir = Path(npy_dir)
    out_bin_path = Path(out_bin_path)
    if out_bin_path.suffix.lower() != ".bin":
        out_bin_path = out_bin_path.with_suffix(".bin")

    files = sorted([p for p in npy_dir.iterdir() if p.is_file() and p.suffix == ".npy"])
    if not files:
        raise RuntimeError(f"目录 {npy_dir} 没有 .npy 切片")

    meta = None
    payload_by_t = {}
    for p in files:
        bits = np.load(p)
        if bits.ndim != 1 or bits.size < HDR_BITS:
            raise ValueError(f"{p.name} 不是合法的一维 bit 切片")

        head_bits = bits[:HDR_BITS]
        header_bytes = np.packbits(head_bits, bitorder=BITORDER).tobytes()
        bps, T, t, H, W = struct.unpack(">BBBBB", header_bytes)

        if meta is None:
            meta = {"bps": bps, "T": T, "H": H, "W": W}
        else:
            if (bps, T, H, W) != (meta["bps"], meta["T"], meta["H"], meta["W"]):
                raise ValueError(f"头不一致：{(bps,T,H,W)} vs {(meta['bps'],meta['T'],meta['H'],meta['W'])}")

        payload_bits = bits[HDR_BITS:]
        need = H * W * bps
        if payload_bits.size < need:
            raise ValueError(f"{p.name} 载荷位数不足：{payload_bits.size} < {need}")
        payload_by_t[int(t)] = payload_bits[:need].astype(np.uint8)

    if meta is None:
        raise RuntimeError("未读到任何切片")
    T, H, W, bps = meta["T"], meta["H"], meta["W"], meta["bps"]

    # 确保 t 覆盖 0..T-1
    missing = sorted(set(range(T)) - set(payload_by_t.keys()))
    if missing:
        raise ValueError(f"缺少切片索引：{missing}")

    # 拼接所有载荷位
    bits_all = np.concatenate([payload_by_t[t] for t in range(T)], axis=0).astype(np.uint8)

    # 打包成字节（不足8倍数补零）
    pad = (-bits_all.size) % 8
    if pad:
        bits_all = np.concatenate([bits_all, np.zeros(pad, dtype=np.uint8)])
    packed = np.packbits(bits_all, bitorder=BITORDER).astype(np.uint8)

    # --- 关键改动：用 BytesIO 得到“NPZ字节流”，再自己写到 .bin ---
    shape = np.array([1, T, H, W], dtype=np.int64)
    bits_arr = np.array(bps, dtype=np.int64)

    out_bin_path.parent.mkdir(parents=True, exist_ok=True)
    bio = io.BytesIO()
    np.savez(bio, packed=packed, shape=shape, bits=bits_arr)  # 写入内存缓冲
    data = bio.getvalue()                                     # 取到 NPZ 的原始字节
    with open(out_bin_path, "wb") as f:          # 关键：传入已打开的文件对象
        np.savez(f, packed=packed, shape=shape, bits=bits_arr)
    print(f"[OK] 已重构 -> {out_bin_path.resolve()}  (shape={(1, T, H, W)}, bps={bps})")
    return {"out_bin": str(out_bin_path.resolve()), "shape": (1, T, H, W), "bps": bps}

def main():
    DO_SPLIT = False   # 需要切片：bin -> 多个 .npy
    DO_JOIN  = True    # 需要还原：多个 .npy -> .bin

    if DO_SPLIT:
        split_bin_to_npy(BIN_INPUT_PATH, NPY_OUTPUT_DIR)
    if DO_JOIN:
        join_npy_to_bin(NPY_INPUT_DIR, BIN_OUTPUT_PATH)

if __name__ == "__main__":
    main()
