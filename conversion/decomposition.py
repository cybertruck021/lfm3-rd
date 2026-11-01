
# Copyright 2026 LFM3-RD Authors. All rights reserved.

"""

Cholesky-whitened SVD decomposition and gauge-invariant ShortConv alignment.

Architectural Note on SwiGLU FFNs vs. Sequence Operators:

- Sequence operators (FullAttention, ShortConv, SlidingWindowAttention) operate directly on the

  residual stream basis (R^2048), which remains aligned across the 5 motifs (layers 2..21).

  Cholesky-whitened SVD + low-rank adapters capture >98.5% of their activation energy (and 100% for KV).

- In contrast, SwiGLU FFN intermediate layers (R^7680) are permutation-invariant across motifs,

  and the multiplicative gate `SiLU(w1(x)) * w3(x)` squares approximation errors. Therefore,

  FFNs are preserved as phase-specific memory modules while sequence operators form the shared core.

"""

import torch

@torch.no_grad()

def fast_svd_ba(M_mat: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    """Computes rank-r factorization M ≈ B @ A via eigendecomposition of the smaller Gram matrix."""

    d_out, d_in = M_mat.shape

    if rank >= min(d_out, d_in):

        # Lossless full-rank representation (used for k_proj / v_proj at rank_kv = 512)

        if d_out <= d_in:

            eye_r = torch.eye(d_out, dtype=M_mat.dtype, device=M_mat.device)

            return eye_r, M_mat.clone(), torch.ones(d_out, device=M_mat.device)

        else:

            eye_r = torch.eye(d_in, dtype=M_mat.dtype, device=M_mat.device)

            return M_mat.clone(), eye_r, torch.ones(d_in, device=M_mat.device)

    if d_out >= d_in:

        evals, V = torch.linalg.eigh(M_mat.T @ M_mat)

        V_r = V[:, -rank:]

        return M_mat @ V_r, V_r.T, evals.clamp(min=0.0)

    else:

        evals, U = torch.linalg.eigh(M_mat @ M_mat.T)

        U_r = U[:, -rank:]

        return U_r, U_r.T @ M_mat, evals.clamp(min=0.0)

@torch.no_grad()

def balance_ba(B_init: torch.Tensor, A_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    """Balances column norms of B with row norms of A so ||B_{:,k}||_2 == ||A_{k,:}||_2."""

    b_f, a_f = B_init.float(), A_init.float()

    b_n = torch.norm(b_f, dim=0, keepdim=True).clamp(min=1e-6)

    a_n = torch.norm(a_f, dim=1, keepdim=True).clamp(min=1e-6)

    sc = torch.sqrt(b_n.T / a_n)

    return (

        (b_f / sc.T).to(torch.bfloat16).contiguous().cpu(),

        (a_f * sc).to(torch.bfloat16).contiguous().cpu(),

    )

@torch.no_grad()

def align_conv_bx_gauge(W_in_list: list[torch.Tensor], S_in: torch.Tensor, D: int) -> list[torch.Tensor]:

    """

    Gauge-invariant sign and scale alignment for Lfm3ShortConv.in_proj ([3D, D] -> B, C, x).

    Since the pre-convolution gate computes `B * x`, the transformation `(g * B) * (g^-1 * x) == B * x`

    is mathematically exact (zero functional change to each layer) while aligning channel signs and scales

    across the 5 motifs prior to extracting W_shared.

    """

    for m in range(len(W_in_list)):

        W_B = W_in_list[m][0:D, :]

        W_x = W_in_list[m][2 * D : 3 * D, :]

        nb = torch.norm(W_B @ S_in, dim=1, keepdim=True).clamp(min=1e-6)

        nx = torch.norm(W_x @ S_in, dim=1, keepdim=True).clamp(min=1e-6)

        alpha = torch.sqrt(nx / nb).clamp(0.5, 2.0)

        W_in_list[m][0:D, :] = W_B * alpha

        W_in_list[m][2 * D : 3 * D, :] = W_x / alpha

    ref_idx = len(W_in_list) // 2

    ref_B_w = W_in_list[ref_idx][0:D, :] @ S_in

    ref_x_w = W_in_list[ref_idx][2 * D : 3 * D, :] @ S_in

    for m in range(len(W_in_list)):

        if m == ref_idx:

            continue

        cur_B_w = W_in_list[m][0:D, :] @ S_in

        cur_x_w = W_in_list[m][2 * D : 3 * D, :] @ S_in

        dot_bx = (cur_B_w * ref_B_w).sum(dim=1, keepdim=True) + (cur_x_w * ref_x_w).sum(dim=1, keepdim=True)

        signs = torch.where(dot_bx < 0, -torch.ones_like(dot_bx), torch.ones_like(dot_bx))

        W_in_list[m][0:D, :] = W_in_list[m][0:D, :] * signs

        W_in_list[m][2 * D : 3 * D, :] = W_in_list[m][2 * D : 3 * D, :] * signs

    return W_in_list

@torch.no_grad()

def decompose_5_motifs_rd(

    W_list: list[torch.Tensor],

    S_chol: torch.Tensor,

    rank: int,

    row_w_list: list[torch.Tensor] | None = None,

    n_iter: int = 4,

) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], float]:

    """

    Alternating Cholesky-whitened low-rank decomposition:

        W_m ≈ W_shared + B_m @ A_m,   m = 0..4

    minimizing E_x[ || (W_m - W_shared - B_m A_m) x ||_2^2 ].

    """

    n_motifs = len(W_list)

    W_sh = torch.stack(W_list, 0).mean(0)

    B_l: list[torch.Tensor | None] = [None] * n_motifs

    A_l: list[torch.Tensor | None] = [None] * n_motifs

    energies = []

    if rank >= min(W_list[0].shape):

        for m in range(n_motifs):

            dW = W_list[m] - W_sh

            B_l[m] = torch.eye(dW.shape[0], dtype=dW.dtype, device=dW.device)

            A_l[m] = dW.clone()

        return W_sh, B_l, A_l, 100.0

    for it in range(n_iter):

        for m in range(n_motifs):

            dW = W_list[m] - W_sh

            rw = None

            if row_w_list is not None:

                rw_v = row_w_list[m].abs().clamp(min=1e-3)

                rw = (rw_v / rw_v.mean()).unsqueeze(1)

                dW = dW * rw

            B_w, A_w, ev = fast_svd_ba(dW @ S_chol, rank)

            B_l[m] = (B_w / rw) if rw is not None else B_w

            A_l[m] = torch.linalg.solve_triangular(S_chol.T, A_w.T, upper=True).T

            if it == n_iter - 1:

                energies.append((ev[-rank:].sum() / (ev.sum() + 1e-8)).item() * 100.0)

        W_sh = torch.stack([W_list[m] - B_l[m] @ A_l[m] for m in range(n_motifs)], 0).mean(0)

    return W_sh, B_l, A_l, sum(energies) / float(n_motifs)
