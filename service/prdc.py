"""
PRDC: Precision, Recall, Density, Coverage — 四维描述子计算

参考 K4 论文 Algorithm 1:
将任意预训练嵌入映射为四维 PRDC 描述子，
基于 k-NN 流形几何统计计算。

此模块独立于原始 K4 代码库，由 K4-service 自行维护。
"""

from __future__ import annotations

import torch
import numpy as np


def compute_prdc(
    E_ref: np.ndarray,
    E_query: np.ndarray,
    k: int = 5,
    device: str = None,
) -> np.ndarray:
    """
    计算 PRDC 四维描述子。

    Args:
        E_ref: 正常日志嵌入，shape (n_ref, d)
        E_query: 待测日志嵌入，shape (n_query, d)
        k: 近邻数量
        device: 计算设备，'cuda' 或 'cpu'

    Returns:
        PRDC 描述子，shape (n_query, 4)
        每列依次为 [Precision, Recall, Density, Coverage]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    E_ref = torch.from_numpy(E_ref).float().to(device)
    E_query = torch.from_numpy(E_query).float().to(device)
    n_ref = E_ref.shape[0]
    n_query = E_query.shape[0]

    if k >= n_ref:
        k = max(1, n_ref - 1)
    k = min(k, n_query)  # cross-distance topk 沿 dim=1 只有 n_query 个元素

    d_ref = torch.cdist(E_ref, E_ref)  # (n_ref, n_ref)
    d_query = torch.cdist(E_query, E_query)  # (n_query, n_query)
    d_cross = torch.cdist(E_query, E_ref)  # (n_query, n_ref)

    # ref 内部每个点的 k-NN 半径
    _, ref_indices = torch.topk(d_ref, k + 1, largest=False)
    ref_indices = ref_indices[:, 1:]  # 排除自身
    r_ref = torch.take_along_dim(d_ref, ref_indices, dim=1).max(dim=1).values  # (n_ref,)

    # query 内部每个点的 k-NN 半径
    _, query_indices = torch.topk(d_query, min(k + 1, n_query), largest=False)
    query_indices = query_indices[:, 1:]  # 排除自身（可能为空）
    r_query = (
        torch.take_along_dim(d_query, query_indices, dim=1).max(dim=1).values
        if query_indices.numel() > 0
        else torch.zeros(n_query, device=device)
    )  # (n_query,)

    # ---------- Precision ----------
    # 对于 query 中每个点 x_j，找其在 ref 中的 k-NN
    _, topk_cross_indices = torch.topk(d_cross, k, largest=False)  # (n_query, k)
    topk_cross_dists = torch.gather(d_cross, 1, topk_cross_indices)  # (n_query, k)
    # 判断每个 k-NN 是否在对应 ref 点的内部球内: dist < r_ref[i]
    r_ref_expanded = r_ref[topk_cross_indices]  # (n_query, k)，每个query的k-NN对应ref点的半径
    precision_mask = topk_cross_dists < r_ref_expanded  # (n_query, k)
    precision = precision_mask.float().mean(dim=1)  # (n_query,)

    # ---------- Recall ----------
    # 对于 ref 中每个点，找其在 query 中的 k-NN
    d_cross_T = d_cross.t()  # (n_ref, n_query)
    _, topk_cross_T_indices = torch.topk(d_cross_T, k, largest=False)  # (n_ref, k)
    topk_cross_T_dists = torch.gather(d_cross_T, 1, topk_cross_T_indices)  # (n_ref, k)
    r_query_expanded = r_query[topk_cross_T_indices]  # (n_ref, k)
    recall_mask_T = topk_cross_T_dists < r_query_expanded  # (n_ref, k)

    # 完全向量化: scatter_add 将每个 ref 点贡献的 1/k 分散到其 query 近邻
    recall_scores = torch.zeros(n_query, device=device)
    # topk_cross_T_indices: (n_ref, k) -> (n_ref * k,)
    # recall_mask_T: (n_ref, k) -> (n_ref * k,)
    # indices 最大值为 n_query-1，scatter_add 只需 n_query 个累加器
    idx_flat = topk_cross_T_indices.view(-1)  # (n_ref * k,)
    mask_flat = recall_mask_T.float().view(-1)  # (n_ref * k,)
    recall_scores.scatter_add_(0, idx_flat, mask_flat)
    recall = recall_scores / k

    # ---------- Density ----------
    # density = k / (sum of k-NN distances to ref)
    sum_kNN_dists = topk_cross_dists.sum(dim=1)  # (n_query,)
    density = k / (sum_kNN_dists + 1e-8)  # (n_query,)
    # 归一化到 [0, 1] 范围
    density = density / (density.max() + 1e-8)

    # ---------- Coverage ----------
    # coverage = indicator(query 的 k-NN 平均距离 < ref 的平均 k-NN 半径)
    mean_kNN_dist = topk_cross_dists.mean(dim=1)  # (n_query,)
    mean_ref_radius = r_ref.mean()
    coverage = (mean_kNN_dist < mean_ref_radius).float()  # (n_query,)

    prdc = torch.stack([precision, recall, density, coverage], dim=1)  # (n_query, 4)
    prdc = prdc.cpu().numpy()
    if prdc.ndim == 1:
        prdc = prdc.reshape(1, -1)  # n_query=1 时 torch.stack 退化为 1D
    return prdc


def compute_prdc_batch(
    E_ref: np.ndarray,
    E_query: np.ndarray,
    k: int = 5,
    batch_size: int = 1000,
    device: str = None,
) -> np.ndarray:
    """
    批量计算 PRDC，用于大规模数据。

    Args:
        E_ref: 正常日志嵌入，shape (n_ref, d)
        E_query: 待测日志嵌入，shape (n_query, d)
        k: 近邻数量
        batch_size: query 批大小
        device: 计算设备

    Returns:
        PRDC 描述子，shape (n_query, 4)
    """
    n_query = E_query.shape[0]
    all_prdc = []

    for i in range(0, n_query, batch_size):
        batch_query = E_query[i:i + batch_size]
        batch_prdc = compute_prdc(E_ref, batch_query, k=k, device=device)
        all_prdc.append(batch_prdc)

    return np.concatenate(all_prdc, axis=0)
