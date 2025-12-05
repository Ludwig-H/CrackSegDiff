import torch
import torch.nn.functional as F

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """
    Pure PyTorch implementation of selective_scan_fn.
    
    Args:
        u: (Batch, Dim, L)
        delta: (Batch, Dim, L)
        A: (Dim, N)
        B: (Batch, K, N, L) or (Batch, N, L)
        C: (Batch, K, N, L) or (Batch, N, L)
        D: (Dim,)
        z: (Batch, Dim, L) - Gate
        delta_bias: (Dim,)
        delta_softplus: bool
        return_last_state: bool
        
    Returns:
        out: (Batch, Dim, L)
    """
    
    # Basic shape checks and adjustments
    batch_size, dim, seq_len = u.shape
    d_state = A.shape[1]
    
    # Handle delta_bias and softplus
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, dim, 1)
        
    if delta_softplus:
        delta = F.softplus(delta)
        
    # Check if B and C are grouped (VMamba case)
    # B input could be (Batch, K, N, L) or (Batch, N, L)
    # We want to broadcast everything to (Batch, Dim, L, N) or (Batch, K, D, L, N)
    
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    
    # Assume VMamba usage pattern:
    # u: (b, k*d, l)
    # B: (b, k, n, l)
    # A: (k*d, n)
    
    if B.dim() == 4: # (b, k, n, l)
        K = B.shape[1]
        assert dim % K == 0
        d_inner = dim // K
        
        # Reshape inputs to (b, k, d, l)
        u_r = u.view(batch_size, K, d_inner, seq_len)
        delta_r = delta.view(batch_size, K, d_inner, seq_len)
        A_r = A.view(K, d_inner, d_state) # (k, d, n)
        
        # B: (b, k, n, l) -> (b, k, 1, l, n) for broadcasting over d
        B_r = B.permute(0, 1, 3, 2).unsqueeze(2) # (b, k, 1, l, n)
        C_r = C.permute(0, 1, 3, 2).unsqueeze(2) # (b, k, 1, l, n)
        
    elif B.dim() == 3: # (b, n, l) - standard mamba
        # Treat as K=1
        K = 1
        d_inner = dim
        
        u_r = u.view(batch_size, 1, d_inner, seq_len)
        delta_r = delta.view(batch_size, 1, d_inner, seq_len)
        A_r = A.view(1, d_inner, d_state)
        
        B_r = B.permute(0, 2, 1).view(batch_size, 1, 1, seq_len, d_state)
        C_r = C.permute(0, 2, 1).view(batch_size, 1, 1, seq_len, d_state)
        
    else:
        raise ValueError(f"Unsupported B shape: {B.shape}")

    # Compute discrete A and B
    # deltaA: exp(delta * A)
    # delta: (b, k, d, l) -> (b, k, d, l, 1)
    # A: (k, d, n) -> (1, k, d, 1, n)
    
    delta_r_uns = delta_r.unsqueeze(-1)
    A_r_uns = A_r.view(1, K, d_inner, 1, d_state)
    
    deltaA = torch.exp(delta_r_uns * A_r_uns) # (b, k, d, l, n)
    
    # deltaB: delta * B * u (kind of, but u is applied later or B is applied to u)
    # Standard: h_t = A_bar * h_{t-1} + B_bar * u_t
    # B_bar approx delta * B
    # In Mamba: B is applied to u first? 
    # Mamba paper: h_t = A_bar h_{t-1} + delta * B * u_t
    # Here B is (b, k, n, l), u is (b, k, d, l)
    # B_r is (b, k, 1, l, n)
    # u_r is (b, k, d, l) -> (b, k, d, l, 1)
    
    u_r_uns = u_r.unsqueeze(-1)
    deltaB_u = delta_r_uns * B_r * u_r_uns # (b, k, d, l, n)
    
    # Recurrent scan
    # h: (b, k, d, n)
    h = torch.zeros(batch_size, K, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    
    # Sequential loop (slow but correct)
    for t in range(seq_len):
        # h = deltaA_t * h + deltaB_u_t
        h = deltaA[:, :, :, t, :] * h + deltaB_u[:, :, :, t, :]
        
        # y_t = C_t * h
        # C_r: (b, k, 1, l, n) -> C_t: (b, k, 1, n)
        # h: (b, k, d, n)
        # sum over n
        y_t = torch.sum(h * C_r[:, :, :, t, :], dim=-1) # (b, k, d)
        ys.append(y_t)
        
    y = torch.stack(ys, dim=-1) # (b, k, d, l)
    
    # Reshape back
    y = y.view(batch_size, dim, seq_len)
    
    # Apply D (skip connection)
    if D is not None:
        y = y + u * D.view(1, dim, 1)
        
    # Apply z (gate)
    if z is not None:
        y = y * F.silu(z)
        
    return y

def selective_scan_ref(*args, **kwargs):
    """
    Reference implementation alias.
    """
    return selective_scan_fn(*args, **kwargs)
