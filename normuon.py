import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl
from collections import defaultdict

# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99

def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def XXT_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )
    return out

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ba_plus_cAA_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )
    return out

# Computed for num_iters=5, safety_factor=2e-2, cushion=2
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def polar_express(G: torch.Tensor, split_baddbmm: bool = False):
    """
    Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
    """
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    # Select batched vs unbatched
    if split_baddbmm:
        BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
    else:
        aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)  # A = X @ X.mT
        ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A

        # Referencing X twice causes pytorch to make a defensive copy,
        # resulting in a cudaMemcpyAsync in baddbmm.
        # For large matrices (i.e., the mlp weights), it's faster to split
        # the operation into two kernels to avoid this.
        if split_baddbmm:
            BX_matmul(B, X, out=C)  # C = B @ X
            C.add_(X, alpha=a)      # C = C + a*X  (in-place, X only read)
        else:
            aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X

        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# -----------------------------------------------------------------------------
# Compiled helpers for NorMuon by @chrisjmccormick

@torch.compile(dynamic=False, fullgraph=True)
def cautious_wd_and_update_inplace(p, v, wd_tensor, lr_tensor):
    """Cautious weight decay + parameter update. wd_tensor and lr_tensor are 0-D CPU tensors."""
    mask = (v * p) >= 0
    wd_factor = wd_tensor.to(p.dtype)
    lr_factor = lr_tensor.to(p.dtype)
    p.copy_(p - (p * mask * wd_factor * lr_factor) - (v * lr_factor))


@torch.compile(dynamic=False, fullgraph=True)
def apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
    """NorMuon variance reduction. Algebraically fuses the normalization steps to minimize memory ops."""
    v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v_chunk.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
    v_norm = v_norm_sq.sqrt_()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
    return v_chunk.mul_(final_scale.type_as(v_chunk))


# -----------------------------------------------------------------------------
# NorMuon optimizer (Single GPU Variant for NeonBench)

class NorMuon(torch.optim.Optimizer):
    """
    Faithful Single-GPU adaptation of the TrainingManager/Optimizers from modded-nanogpt.
    
    This optimizer routes parameters into three separate groups based on their `.label` attribute
    (or dimensions as a fallback):
    1. Adam params (embeddings, small gates) -> lr=0.004, wd=0.005, betas=(0.8, 0.95), odd_step_only=True
    2. Scalar params (residuals) -> lr=0.008, wd=0.005, betas=(0.9, 0.99), odd_step_only=True
    3. Muon params (2D matrices) -> lr=0.015, wd=1.2, momentum=0.95, beta2=0.95
    """
    def __init__(self, params, lr=1.0): # lr is ignored, used as placeholder for train.py compat
        defaults = dict(lr=lr)
        
        adam_labels = ['lm_head', 'value_embed', 'smear_gate', 'skip_gate', 'embed2', 'embed', 'x0_lambdas']
        scalar_labels = ['scalars']
        muon_labels = ['attn_gate', 'value_embed_gate', 'attn', 'mlp', 'conv_q', 'conv_k', 'conv_v', 'conv_i']
        
        adam_params, scalar_params, muon_params = [], [], []
        
        for param in params:
            label = getattr(param, 'label', None)
            if label in adam_labels: adam_params.append(param)
            elif label in scalar_labels: scalar_params.append(param)
            elif label in muon_labels: muon_params.append(param)
            else:
                # Fallback for models like neon185
                if param.ndim >= 2 and not isinstance(param, nn.Embedding):
                    muon_params.append(param)
                else:
                    adam_params.append(param)
                    
        groups = [
            dict(params=adam_params, optim_type='adam', initial_lr=0.004, betas=(0.8, 0.95), weight_decay=0.005, odd_step_only=True),
            dict(params=scalar_params, optim_type='adam', initial_lr=0.008, betas=(0.9, 0.99), weight_decay=0.005, odd_step_only=True),
            dict(params=muon_params, optim_type='normuon', initial_lr=0.015, momentum=0.95, beta2=0.95, weight_decay=1.2, odd_step_only=False)
        ]
        
        super().__init__(groups, defaults)
        self.step_cnt = 0

    def reset(self):
        for group in self.param_groups:
            if "momentum_buffer" in group:
                for mb in group["momentum_buffer"].values():
                    if mb is not None: mb.zero_()
            if "second_momentum_buffer" in group:
                for smb in group["second_momentum_buffer"].values():
                    if smb is not None: smb.zero_()
            if "state" in group:
                for state in group["state"].values():
                    state["step"] = 0
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()

    @torch.no_grad()
    def step(self):
        self.step_cnt += 1
        for group in self.param_groups:
            if group.get("odd_step_only", False) and self.step_cnt % 2 == 0:
                continue
                
            if group['optim_type'] == 'normuon':
                self._step_normuon(group)
            elif group['optim_type'] == 'adam':
                self._step_adam(group)

    @torch.no_grad()
    def _step_normuon(self, group):
        params: list[Tensor] = group["params"]
        if not params: return

        if "momentum_buffer" not in group:
            group["momentum_buffer"] = {p: torch.zeros_like(p) for p in params}
        if "second_momentum_buffer" not in group:
            group["second_momentum_buffer"] = {}
            for p in params:
                shape = p.shape
                if shape[-2] >= shape[-1]: group["second_momentum_buffer"][p] = torch.zeros_like(p[..., :, :1])
                else: group["second_momentum_buffer"][p] = torch.zeros_like(p[..., :1, :])
                    
        if "param_lr_cpu" not in group:
            group["param_lr_cpu"] = {}
            group["param_wd_cpu"] = {}
            for p in params:
                shape = p.shape
                shape_mult = max(1.0, shape[-2] / shape[-1]) ** 0.5 if len(shape) >= 2 else 1.0
                group["param_lr_cpu"][p] = torch.tensor(shape_mult * getattr(p, "lr_mul", 1.0), dtype=torch.float32, device="cpu")
                group["param_wd_cpu"][p] = torch.tensor(getattr(p, "wd_mul", 1.0), dtype=torch.float32, device="cpu")

        for param in params:
            if param.grad is None: continue
            
            grad = param.grad
            momentum_buffer = group["momentum_buffer"][param]
            second_momentum_buffer = group["second_momentum_buffer"][param]
            
            momentum_buffer.lerp_(grad, 1 - group.get("momentum", 0.95))
            updated_grad = grad.lerp_(momentum_buffer, group.get("momentum", 0.95))
            
            red_dim = -1 if param.shape[-2] >= param.shape[-1] else -2
            v = polar_express(updated_grad, split_baddbmm=False)
            v = apply_normuon_variance_reduction(v, second_momentum_buffer, group["beta2"], red_dim)
            v = v.view(param.shape)
            
            eff_lr_cpu = group["param_lr_cpu"][param] * group["lr"]
            eff_wd_cpu = group["param_wd_cpu"][param] * group["weight_decay"] * group["lr"]
            
            cautious_wd_and_update_inplace(param, v, eff_wd_cpu, eff_lr_cpu)

    @torch.no_grad()
    def _step_adam(self, group):
        params: list[Tensor] = group["params"]
        if not params: return
            
        if "state" not in group:
            group["state"] = {}
            for p in params:
                group["state"][p] = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p)
                }
                
        beta1, beta2 = group["betas"]
        eps = group.get("eps", 1e-8)
        wd = group["weight_decay"]
        lr = group["lr"]

        for param in params:
            if param.grad is None: continue
            
            grad = param.grad
            state = group["state"][param]
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            t = state["step"]
            
            # Use specific lr_mul and wd_mul like modded-nanogpt
            p_lr = lr * getattr(param, "lr_mul", 1.0)
            p_wd = wd * getattr(param, "wd_mul", 1.0)
            
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            bias1 = 1 - beta1 ** t
            bias2 = 1 - beta2 ** t
            
            denom = exp_avg_sq.sqrt().add_(eps)
            step_size = p_lr * (bias2 ** 0.5 / bias1)
            update = exp_avg.div(denom).mul_(step_size)
            
            # Cautious weight decay for Adam too, like modded-nanogpt!
            mask = (update * param) > 0
            eff_weight_decay = p_lr * p_wd
            update.addcmul_(param, mask, value=eff_weight_decay)
            
            param.add_(other=update, alpha=-1.0)

# -----------------------------------------------------------------------------
# Muon - MomentUm Orthogonalized by Polar Express / Newton Schulz
# From the provided reference repository

@torch.compile()
def zeropower_polar_express(G: torch.Tensor, steps: int = 5):
    """Polar express as replacement for Newton-Schulz iteration"""
    assert G.ndim >= 2
    assert steps <= len(polar_express_coeffs)

    X = G.bfloat16()
    
    transpose_needed = G.size(-2) > G.size(-1) # transposing if tall matrix
    if transpose_needed: 
        X = X.mT 
    
    # Using the safety factor from the reference repo
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7) 
    
    coeffs = polar_express_coeffs[:steps]
    for a, b, c in coeffs:
        A = X @ X.mT 
        A2 = A @ A 
        B = b * A + c * A2
        X = a * X + B @ X  # Right-multiplication for left polar factor
    
    if transpose_needed: 
        X = X.mT 
    
    return X # orthogonalized 

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Polar Express / Newton Schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_polar_express(g, steps=group["ns_steps"]) 
                g = g.to(p.dtype)
                
                # Using the scaling from the reference repo
                scaling = max(1, p.size(-2) / p.size(-1))**0.5
                p.add_(g.view_as(p), alpha=-group["lr"] * scaling)
        
        return loss
class MuonGatedAdam(torch.optim.Optimizer):
    """
    Muon-Gated Adam Optimizer:
    Uses Muon (orthogonalized momentum) as a spectral gate for AdamW updates.
    Gated Update = AdamW_Update * Muon_Orthogonal_Update
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, 
                 muon_momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, 
                        muon_momentum=muon_momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            muon_mom = group["muon_momentum"]
            
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(g)
                    state["exp_avg_sq"] = torch.zeros_like(g)
                    state["muon_buf"] = torch.zeros_like(g)

                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                muon_buf = state["muon_buf"]
                
                # 1. Standard AdamW Update Calculation
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                
                bias_corr1 = 1 - beta1 ** state["step"]
                bias_corr2 = 1 - beta2 ** state["step"]
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(group["eps"])
                # Normalized Adam Update (spectral "magnitude" is not unit, but coordinate "magnitude" is stabilized)
                g_adam = (exp_avg / bias_corr1) / denom 
                
                # 2. Muon Gate Calculation
                muon_buf.lerp_(g, 1 - muon_mom)
                # Polar Express orthogonalizes the momentum buffer
                # For Gated Muon, we treat this as the "Directional Mask"
                g_muon = zeropower_polar_express(muon_buf, steps=group["ns_steps"])
                g_muon = g_muon.to(p.dtype)
                
                # 3. Gated Application: Update = Adam * Muon
                # Scaling factor for Muon usually accounts for aspect ratio
                scaling = max(1, p.size(-2) / p.size(-1))**0.5
                
                # Element-wise gate
                actual_update = g_adam * g_muon.view_as(g_adam)
                
                # Apply weight decay (AdamW style)
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                
                # Update parameters
                p.add_(actual_update, alpha=-group["lr"] * scaling)

        return loss
