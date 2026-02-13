"""NeonBench Model Visualizer — Interactive attention & vector heatmaps.
Run: streamlit run scripts/visualizer.py
"""
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys, importlib, json
from tokenizers import Tokenizer

# --- Setup ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from train import get_config

CKPT_DIR = os.path.join(ROOT, "checkpoints")
TOK_DIR = os.path.join(ROOT, "tokenizers")

st.set_page_config(page_title="NeonBench Visualizer", layout="wide")

# ── Helpers ──────────────────────────────────────────────────────────

def scan_checkpoints():
    """Return dict of label -> {path, model_name, tok_name, data_name}."""
    ckpts = {}
    if not os.path.isdir(CKPT_DIR):
        return ckpts
    for f in sorted(os.listdir(CKPT_DIR)):
        if not f.endswith("_best.pth"):
            continue
        stem = f.replace("_best.pth", "")
        parts = stem.split("_")
        model = parts[0]
        tok = parts[1] if len(parts) >= 2 else "tok1"
        data = "_".join(parts[2:]) if len(parts) >= 3 else "hp0"
        label = f"{model}  ({tok} / {data})"
        ckpts[label] = dict(path=os.path.join(CKPT_DIR, f),
                            model_name=model, tok_name=tok, data_name=data)
    return ckpts

def find_tokenizer(tok_name, data_name):
    """Find a tokenizer file that strictly matches both tok_name and data_name."""
    # 1. Try exact dataset + tok match (e.g., wiki103_tok4.json)
    p = os.path.join(TOK_DIR, f"{data_name}_{tok_name}.json")
    if os.path.exists(p):
        return p
    
    # 2. Handle 'hp0' -> 'hp' alias for directory consistency
    data_alias = data_name.replace("hp0", "hp")
    p = os.path.join(TOK_DIR, f"{data_alias}_{tok_name}.json")
    if os.path.exists(p):
        return p

    # 3. Try just the data name if the tok_name is embedded (e.g., wiki103.json)
    p = os.path.join(TOK_DIR, f"{data_name}.json")
    if os.path.exists(p):
        return p

    # 4. Fallback to listdir but insist on data_name (or alias) being in the string
    for f in os.listdir(TOK_DIR):
        if f.endswith(f"_{tok_name}.json") and (data_name in f or data_alias in f):
            return os.path.join(TOK_DIR, f)
            
    return None

@st.cache_resource
def load_model(model_name, ckpt_path, tok_name, data_name):
    try:
        tok_path = find_tokenizer(tok_name, data_name)
        if tok_path is None:
            return None, None, None
        
        with open(tok_path, 'r', encoding='utf-8') as f:
            tok_data = json.load(f)
        
        if tok_data.get('type') == 'word_level_pos':
            from scripts.build_warm_tokenizer import WarmTokenizer
            tokenizer = WarmTokenizer(tok_path)
            vocab_size = len(tokenizer)
        else:
            tokenizer = Tokenizer.from_file(tok_path)
            vocab_size = tokenizer.get_vocab_size()

        config = get_config(model_name)
        config['vocab_size'] = vocab_size

        cls_name = model_name.capitalize()
        mod = importlib.import_module(f"models.{model_name}")
        ModelClass = getattr(mod, cls_name)
        model = ModelClass(config)
        
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, tokenizer, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def clean_token(t):
    """Make BPE tokens more readable."""
    return t.replace("Ġ", "·").replace("Ċ", "↵").replace("▁", "·")

def token_labels(tokens):
    """Create unique labels: prepend position index to avoid Plotly duplicate-label issues."""
    return [f"{i}:{clean_token(t)}" for i, t in enumerate(tokens)]


# ── Attention + MLP + Vector capture ─────────────────────────────────

def capture_forward(model, input_ids):
    """Run forward pass, capture Q, K, V, Intent, Attention Weights, and MLP activations."""
    
    # Storage buckets
    attn_bucket = []
    q_bucket = []
    k_bucket = []
    v_bucket = []
    intent_bucket = []  # Stores (B, n_head, T, head_dim)
    mlp_bucket = []
    conv_bucket = [] # Stores convolutional intermediate feature maps
    gate_bucket = [] # Stores the final multiplication gate

    # Original functions
    real_sdpa = F.scaled_dot_product_attention
    real_sigmoid = torch.sigmoid
    real_silu = F.silu

    # 1. SDPA Hook (Captures Q, K, V, Attn)
    def spy_sdpa(q, k, v, *args, **kwargs):
        # Capture Vectors (detach to save memory)
        q_bucket.append(q.detach().cpu())
        k_bucket.append(k.detach().cpu())
        v_bucket.append(v.detach().cpu())

        # Compute Attention Weights (Replicate logic roughly to get weights)
        L, S = q.size(-2), k.size(-2)
        scale = kwargs.get('scale', None)
        s = 1.0 / (q.size(-1) ** 0.5) if scale is None else scale
        logits = q @ k.transpose(-2, -1) * s
        mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=q.device), diagonal=1)
        logits.masked_fill_(mask, float("-inf"))
        w = torch.softmax(logits, dim=-1)
        attn_bucket.append(w.detach().cpu())
        
        # We must return the Output of SDPA (V weighted by attn)
        # Note: If model does subsequent gating (neon016), it happens AFTER this returns.
        return w @ v

    # 2. Intent Hook (Captures Sigmoid/SiLU inputs)
    def spy_sigmoid(input):
        if input.dim() in [3, 4]: # Support both MQI (3D) and MHI/Multi-Head (4D)
            intent_bucket.append(input.detach().cpu())
        return real_sigmoid(input)

    def spy_silu(input, inplace=False):
        if input.dim() in [3, 4]:
            intent_bucket.append(input.detach().cpu())
        return real_silu(input, inplace=inplace)

    # 3. Hydra Convolution Hook (Monkey-patch Conv1d)
    real_conv1d = F.conv1d
    def spy_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        out = real_conv1d(input, weight, bias, stride, padding, dilation, groups)
        if groups > 1 and out.dim() == 3: # Likely a Depthwise/Hydra conv
            conv_bucket.append({
                'kernel': weight.shape[2],
                'data': out.detach().cpu()
            })
        return out

    # 3. MLP Hook
    hooks = []
    for block in model.blocks:
        if hasattr(block, 'mlp'):
            def make_hook():
                def hook_fn(module, inp, out):
                    # inp[0] is the input to the down-proj, which is the hidden state (B, T, d_ff)
                    # OR if we hooked the MLP block, inp is (B, T, d_model) and out is (B, T, d_model)
                    # We prefer inner hidden state if possible.
                    data = inp[0].detach().cpu()
                    mlp_bucket.append({
                        'input': data,  # This will be d_ff if hooked on down_proj
                        'output': out.detach().cpu(),
                    })
                return hook_fn
            
            # Detect architecture style
            # neon015/055 (SwiGLU) uses 'w_down'
            # neon001 (GPT-2) uses 'c_proj'
            # Hydra (070+) uses 'w2'
            target_layer = getattr(block.mlp, 'w2', getattr(block.mlp, 'w_down', getattr(block.mlp, 'c_proj', block.mlp)))
            
            hooks.append(target_layer.register_forward_hook(make_hook()))

    # Apply Patches
    F.scaled_dot_product_attention = spy_sdpa
    torch.sigmoid = spy_sigmoid
    F.silu = spy_silu
    F.conv1d = spy_conv1d
    
    try:
        with torch.no_grad():
            logits, _ = model(input_ids)
            last_logits = logits[0, -1].detach().cpu()
    finally:
        # Restore
        F.scaled_dot_product_attention = real_sdpa
        torch.sigmoid = real_sigmoid
        F.silu = real_silu
        F.conv1d = real_conv1d
        for h in hooks:
            h.remove()
            
    return {
        "attn": attn_bucket,
        "q": q_bucket,
        "k": k_bucket,
        "v": v_bucket,
        "intent": intent_bucket,
        "mlp": mlp_bucket,
        "conv": conv_bucket,
        "last_logits": last_logits
    }

@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=50):
    """Simple generation loop for the visualizer."""
    model.eval()
    device = next(model.parameters()).device
    
    # Encode
    if hasattr(tokenizer, 'encode'):
        res = tokenizer.encode(prompt)
        # Handle BPE vs Warm vs List
        if hasattr(res, 'ids'): ids = res.ids
        elif isinstance(res, list): ids = res
        else: ids = res.ids # Fallback for other tokenizer wrappers
    else:
        ids = tokenizer.encode(prompt)
        
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    block_size = model.config['block_size']
    
    placeholder = st.empty()
    
    for _ in range(max_new_tokens):
        # Crop
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        # Forward
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        
        # Top-k
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append
        idx = torch.cat([idx, next_token], dim=1)
        
        # Decode and show
        full_ids = idx[0].tolist()
        try:
            current_text = tokenizer.decode(full_ids)
        except Exception:
            current_text = str(full_ids) # Fallback
            
        placeholder.markdown(current_text + " ▌")
        
        # Stop check
        if next_token.item() == 0: # Usually [PAD] or something
             break
            
    placeholder.markdown(current_text)
    return current_text


# ── Plotting ─────────────────────────────────────────────────────────

def plot_single_head(attn_matrix, tokens, layer, head):
    """Interactive plotly heatmap for one attention head."""
    labels = token_labels(tokens)
    T = len(labels)
    fig = go.Figure(go.Heatmap(
        z=attn_matrix,
        x=labels, y=labels,
        colorscale="Blues",
        zmin=0, zmax=float(attn_matrix.max()),
        hoverongaps=False,
        xgap=1, ygap=1,
        text=np.round(attn_matrix, 3),
        texttemplate="%{text}" if T <= 25 else "",
        textfont_size=9,
    ))
    fig.update_layout(
        title=f"Layer {layer} — Head {head} (Attention Weights)",
        xaxis_title="Key (attended to →)",
        yaxis_title="Query (attending from ↓)",
        yaxis=dict(autorange="reversed"),
        height=max(420, T * 22 + 120),
        margin=dict(l=80, r=20, t=50, b=80),
    )
    fig.update_xaxes(tickangle=45)
    return fig

def plot_vector_heatmap(vector_data, tokens, layer, head, component_name, n_head_expected):
    """Heatmap for Q, K, V, or Intent vectors. Shape (T, head_dim)."""
    # vector_data[layer] is (B, D1, D2, D3). Squeeze batch.
    tensor = vector_data[layer][0]
    
    # Try to find the head axis. Common layouts: (n_head, T, D) or (T, n_head, D)
    if tensor.shape[0] == n_head_expected:
        data = tensor[head].numpy()
    elif tensor.shape[1] == n_head_expected:
        data = tensor[:, head].numpy()
    else:
        # Fallback
        data = tensor[head].numpy()
    
    labels = token_labels(tokens)
    T, D = data.shape
    
    # Auto-scale colors (centered at 0 for vectors)
    mx = abs(data).max()
    
    fig = go.Figure(go.Heatmap(
        z=data,
        x=[f"d{i}" for i in range(D)],
        y=labels,
        colorscale="RdBu", 
        zmin=-mx, zmax=mx,
        xgap=1, ygap=0, # Gap between columns, but dense rows
    ))
    fig.update_layout(
        title=f"Layer {layer} — Head {head} ({component_name})",
        xaxis_title="Dimension",
        yaxis_title="Token",
        yaxis=dict(autorange="reversed"),
        height=max(420, T * 22 + 120),
        margin=dict(l=80, r=20, t=50, b=80),
    )
    return fig

def plot_all_heads(attn_layer, tokens, layer, n_heads):
    """Small-multiple grid of all heads for one layer."""
    cols = min(n_heads, 4)
    rows = (n_heads + cols - 1) // cols
    labels = token_labels(tokens)

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Head {h}" for h in range(n_heads)],
                        horizontal_spacing=0.04, vertical_spacing=0.08)
    for h in range(n_heads):
        r, c = divmod(h, cols)
        fig.add_trace(go.Heatmap(
            z=attn_layer[0, h].numpy(),
            x=labels, y=labels,
            colorscale="Blues", showscale=(h == 0),
            zmin=0, zmax=float(attn_layer[0].max()),
            xgap=1, ygap=1,
        ), row=r + 1, col=c + 1)
        fig.update_yaxes(autorange="reversed", row=r + 1, col=c + 1)

    T = len(labels)
    fig.update_layout(
        title=f"All Heads — Layer {layer}",
        height=max(350, T * 18 + 80) * rows,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def plot_attention_received(attn_matrix, tokens, layer, head):
    """Bar chart: total attention each token receives (column sum)."""
    labels = token_labels(tokens)
    received = attn_matrix.sum(axis=0)
    fig = go.Figure(go.Bar(x=labels, y=received, marker_color="#1f77b4"))
    fig.update_layout(
        title=f"Attention Received — Layer {layer}, Head {head}",
        xaxis_title="Token", yaxis_title="Total attention received",
        height=280, margin=dict(l=60, r=20, t=40, b=60),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_full_grid(attns, tokens, n_layers, n_heads):
    """Full grid: rows=layers, cols=heads."""
    labels = token_labels(tokens)
    fig = make_subplots(
        rows=n_layers, cols=n_heads,
        subplot_titles=[f"L{l} H{h}" for l in range(n_layers) for h in range(n_heads)],
        horizontal_spacing=0.02, vertical_spacing=0.03,
    )
    global_max = max(float(a[0].max()) for a in attns)
    for l in range(n_layers):
        for h in range(n_heads):
            fig.add_trace(go.Heatmap(
                z=attns[l][0, h].numpy(),
                x=labels, y=labels,
                colorscale="Blues", showscale=False,
                zmin=0, zmax=global_max,
                xgap=1, ygap=1,
            ), row=l + 1, col=h + 1)
            fig.update_yaxes(autorange="reversed", showticklabels=False, row=l + 1, col=h + 1)
            fig.update_xaxes(showticklabels=False, row=l + 1, col=h + 1)
    T = len(labels)
    cell_h = max(120, T * 10 + 30)
    fig.update_layout(
        title="All Layers × All Heads",
        height=cell_h * n_layers + 60,
        margin=dict(l=30, r=20, t=60, b=30),
    )
    for l in range(n_layers):
        fig.add_annotation(text=f"L{l}", x=-0.02, y=1 - (l + 0.5) / n_layers,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=11, color="gray"))
    return fig


def plot_mlp_activation(mlp_data, tokens, layer):
    """Heatmap of MLP hidden state magnitude per token per dimension."""
    # Start with input to the down-projection (hidden state)
    data = mlp_data[layer]['input'][0].numpy()  # (T, d_ff)
    
    # If the hook caught the block wrapper, 'input' is d_model. Check dims.
    T, D = data.shape
    
    # User requested seeing all dimensions (e.g. 592)
    # No sorting/filtering.
    
    labels = token_labels(tokens)
    
    # Auto-scale colors
    mx = abs(data).max()
    
    fig = go.Figure(go.Heatmap(
        z=data.T,
        x=labels,
        y=[f"d{d}" for d in range(D)],
        colorscale="RdBu_r", zmid=0, zmin=-mx, zmax=mx,
        xgap=1, ygap=0,
    ))
    fig.update_layout(
        title=f"MLP Hidden State — Layer {layer} ({D} dimensions)",
        xaxis_title="Token", yaxis_title="Dimension",
        height=max(500, D * 10 + 100), # Ensure enough height if D is large
        margin=dict(l=60, r=20, t=50, b=80),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_mlp_norm(mlp_data, tokens, n_layers):
    """Heatmap: L2 norm of MLP output per token per layer."""
    labels = token_labels(tokens)
    norms = []
    for l in range(n_layers):
        out = mlp_data[l]['output'][0].numpy()  # (T, d_model)
        norms.append(np.linalg.norm(out, axis=1))  # (T,)
    norms = np.array(norms)  # (n_layers, T)
    fig = go.Figure(go.Heatmap(
        z=norms, x=labels,
        y=[f"Layer {l}" for l in range(n_layers)],
        colorscale="Oranges", xgap=1, ygap=1,
    ))
    fig.update_layout(
        title="MLP Output Norm — All Layers",
        xaxis_title="Token", yaxis_title="Layer",
        height=max(250, n_layers * 50 + 100),
        margin=dict(l=80, r=20, t=50, b=80),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_next_token_probs(logits, tokenizer, top_k=20):
    """Horizontal bar chart of top-K next token probabilities."""
    probs = torch.softmax(logits, dim=-1).numpy()
    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    
    # Decode labels
    labels = []
    for idx in top_indices:
        try:
            # Handle diff tokenizer types
            if hasattr(tokenizer, 'decode'):
                t = tokenizer.decode([int(idx)])
            else:
                t = f"id:{idx}"
            labels.append(f"`{clean_token(t)}`")
        except:
            labels.append(f"id:{idx}")

    fig = go.Figure(go.Bar(
        x=top_probs,
        y=labels,
        orientation='h',
        marker_color='rgb(55, 83, 109)'
    ))
    fig.update_layout(
        title=f"Top {top_k} Next Token Probabilities",
        xaxis_title="Probability",
        yaxis=dict(autorange="reversed"),
        height=max(400, top_k * 25),
        margin=dict(l=150, r=20, t=50, b=50),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════

st.title("🔬 NeonBench Attention Visualizer")

# ── Sidebar ──
st.sidebar.header("Model")
checkpoints = scan_checkpoints()
if not checkpoints:
    st.error("No checkpoints found in `checkpoints/`.  \n"
             "Pull from server: `git pull`")
    st.stop()

selected = st.sidebar.selectbox("Checkpoint", list(checkpoints.keys()))
info = checkpoints[selected]

model, tokenizer, config = load_model(
    info['model_name'], info['path'], info['tok_name'], info['data_name'])

if model is None:
    st.error(f"❌ **Load Failed** for `{selected}`")
    st.info(f"Looking for tokenizer matching `{info['data_name']}` + `{info['tok_name']}`")
    st.stop()

n_layers = config['n_layers']
n_heads  = config['n_head']
n_params = sum(p.numel() for p in model.parameters())

st.sidebar.markdown(f"""
| | |
|---|---|
| **Architecture** | `{info['model_name']}` |
| **Layers** | {n_layers} |
| **Heads** | {n_heads} |
| **d_model** | {config['d_model']} |
| **d_ff** | {config['d_ff']} |
| **Parameters** | {n_params:,} |
| **Trained on** | `{info['data_name']}` |
| **Tokenizer** | `{os.path.basename(tk_path) if (tk_path := find_tokenizer(info['tok_name'], info['data_name'])) else 'Not Found'}` |
""")

# ── Prompt input ──
prompt = st.text_area("✏️ Enter a prompt:",
                      value="Harry looked at the mirror and saw",
                      height=80)

if st.button("🔍 Visualize", type="primary") or "data" in st.session_state:
    # Only recompute if prompt changed or first run
    current_key = f"{selected}|{prompt}"
    if st.session_state.get("_vis_key") != current_key or "data" not in st.session_state:
        encoded = tokenizer.encode(prompt)
        if hasattr(encoded, 'ids'):
            ids = encoded.ids
            toks = encoded.tokens
        else:
            # WarmTokenizer or list-style
            ids = encoded
            toks = [tokenizer.decode([i]) for i in ids]

        if not ids:
            st.error("Tokenizer produced no tokens. Try a different prompt.")
            st.stop()
        if len(ids) > config['block_size']:
            st.warning(f"Truncated to {config['block_size']} tokens.")
            ids = ids[:config['block_size']]
            toks = toks[:config['block_size']]

        input_tensor = torch.tensor([ids])
        data = capture_forward(model, input_tensor)

        st.session_state["data"] = data
        st.session_state["tokens"] = toks
        st.session_state["_vis_key"] = current_key

    data = st.session_state["data"]
    toks = st.session_state["tokens"]
    
    attns = data["attn"]
    mlps  = data["mlp"]
    
    st.success(f"**{len(toks)} tokens** × {n_layers} layers × {n_heads} heads")
    st.markdown("**Tokens:** " + "  ".join(f"`{clean_token(t)}`" for i, t in enumerate(toks)))

    # ── Tabs ──
    tabs = st.tabs(["🔎 Single Head", "📊 All Heads", "🧩 Full Grid",
                    "📈 Attention Flow", "🔧 MLP Activations", "🔮 Next Token", "💬 Chat/Inference"])

    # --- Single Head ---
    with tabs[0]:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            layer = st.selectbox("Layer", range(n_layers),
                                 format_func=lambda x: f"Layer {x}", key="sh_layer")
        with c2:
            head = st.selectbox("Head", range(n_heads),
                                format_func=lambda x: f"Head {x}", key="sh_head")
                                
        # Component Selector
        components = ["Attention Matrix (T×T)", "Query Vector (T×D)", "Key Vector (T×D)", "Value Vector (T×D)"]
        if data["intent"]:
            components.append("Intent Vector (T×D)")
        
        # Hydra specific components
        if data["conv"]:
            components.append("Conv Gate (Feature Map)")
            if len(data["conv"]) > n_layers: 
                components.append("MLP Conv Gate (Feature Map)")
            
        with c3:
            comp = st.selectbox("Component", components, key="sh_comp")
            
        if "Attention" in comp:
            if attns and layer < len(attns):
                attn_np = attns[layer][0, head].numpy()
                st.plotly_chart(plot_single_head(attn_np, toks, layer, head),
                                use_container_width=True)
                st.plotly_chart(plot_attention_received(attn_np, toks, layer, head),
                                use_container_width=True)
            else:
                st.warning("⚠️ This model does not use Attention to compute this layer. Select 'Conv Gate' or 'MLP' instead.")
        elif "Conv" in comp:
            # Silent Hydra / Attention-Free models have different conv counts
            # We'll try to find the match based on layer index.
            # LocalGatedBlock often has 2 convs (v, g), MLP has 1 (conv9).
            # We look for a conv that matches the requested block type.
            
            is_mlp_view = "MLP" in comp
            
            # Simple heuristic mapping for the "Conv Gate" tab
            # In neon143: 2 convs in Gate block, 1 in MLP.
            # layer 0: Gate-ConvV (idx0), Gate-ConvG (idx1), MLP-Conv9 (idx2)
            # layer 1: Gate-ConvV (idx3), Gate-ConvG (idx4), MLP-Conv9 (idx5)
            
            convs_per_layer = len(data["conv"]) // n_layers
            if is_mlp_view:
                # MLP conv is usually the last in each block's sequence
                idx = (layer + 1) * convs_per_layer - 1 
            else:
                # Gate conv is usually early in the sequence
                idx = layer * convs_per_layer
            
            if 0 <= idx < len(data["conv"]):
                conv_info = data["conv"][idx]
                conv_vals = conv_info['data'][0].numpy()
                k = conv_info.get('kernel', '?')
                
                fig = go.Figure(go.Heatmap(
                    z=conv_vals.T, 
                    x=[f"d{i}" for i in range(conv_vals.shape[0])],
                    y=token_labels(toks),
                    colorscale="Viridis",
                    xgap=0, ygap=0,
                ))
                fig.update_layout(
                    title=f"Layer {layer} — {comp} (k={k}) Feature Map", 
                    height=max(420, len(toks) * 22 + 120),
                    yaxis=dict(autorange="reversed"),
                    xaxis_title="Dimension (Gates)",
                    yaxis_title="Token"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Convolution data mapped for this layer block.")
        else:
            # Map selection to bucket
            if "Query" in comp: vec_data = data["q"]
            elif "Key" in comp: vec_data = data["k"]
            elif "Value" in comp: vec_data = data["v"]
            elif "Intent" in comp: vec_data = data["intent"]
            
            if vec_data:
                st.plotly_chart(plot_vector_heatmap(vec_data, toks, layer, head, comp, n_heads),
                                use_container_width=True)
            else:
                st.warning(f"No data captured for {comp}")

    # --- All Heads (one layer) ---
    with tabs[1]:
        if attns:
            layer_all = st.selectbox("Layer", range(n_layers),
                                     format_func=lambda x: f"Layer {x}", key="ah_layer")
            st.plotly_chart(plot_all_heads(attns[layer_all], toks, layer_all, n_heads),
                            use_container_width=True)
        else:
            st.info("💡 **Attention-Free Model**: This architecture does not use multiple attention heads.")

    # --- Full Grid (all layers × all heads) ---
    with tabs[2]:
        if attns:
            st.markdown(f"**{n_layers} layers × {n_heads} heads** — rows = layers, columns = heads")
            st.plotly_chart(plot_full_grid(attns, toks, n_layers, n_heads),
                            use_container_width=True)
        else:
            st.info("💡 **Attention-Free Model**: No global attention grid to display.")

    # --- Attention Flow ---
    with tabs[3]:
        if attns:
            st.markdown("**Average attention per layer** (mean across all heads)")
            layer_flow = st.selectbox("Layer", range(n_layers),
                                      format_func=lambda x: f"Layer {x}", key="af_layer")
            avg = attns[layer_flow][0].mean(dim=0).numpy()
            labels = token_labels(toks)
            fig = go.Figure(go.Heatmap(
                z=avg, x=labels, y=labels,
                colorscale="Viridis", xgap=1, ygap=1,
                zmin=0, zmax=float(avg.max()),
            ))
            fig.update_layout(
                title=f"Average Attention — Layer {layer_flow}",
                yaxis=dict(autorange="reversed"),
                height=max(420, len(toks) * 22 + 120),
                xaxis_title="Key", yaxis_title="Query",
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 **Gated Convolutional Flow**: In 'Silent' models, signal flow is determined by gating mechanisms rather than attention dot-products.")

    # --- MLP Activations ---
    with tabs[4]:
        if not mlps:
            st.info("No MLP data captured. Model may not have standard MLP blocks.")
        else:
            st.markdown("**MLP output norm across all layers and tokens**")
            st.plotly_chart(plot_mlp_norm(mlps, toks, n_layers),
                            use_container_width=True)
            st.divider()
            mlp_layer = st.selectbox("Layer", range(n_layers),
                                     format_func=lambda x: f"Layer {x}", key="mlp_layer")
            st.plotly_chart(plot_mlp_activation(mlps, toks, mlp_layer),
                            use_container_width=True)

    # --- Next Token Probs ---
    with tabs[5]:
        st.subheader("Next Token Prediction")
        st.markdown("The model's probability distribution for the token immediately following your prompt.")
        top_k_viz = st.slider("Show Top K tokens", 5, 50, 20)
        st.plotly_chart(plot_next_token_probs(data["last_logits"], tokenizer, top_k_viz),
                        use_container_width=True)

    # --- Chat/Inference ---
    with tabs[6]:
        st.subheader("Talk to the Model")
        chat_prompt = st.text_input("Continue from prompt or ask something:", value=prompt, key="chat_input")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            max_tokens = st.slider("Max new tokens", 10, 500, 100)
        with c2:
            temp = st.slider("Temperature", 0.1, 2.0, 0.8)
        with c3:
            top_k = st.slider("Top-k", 1, 100, 50)
            
        if st.button("🚀 Generate Response", type="primary"):
            with st.spinner("Model is thinking..."):
                generate_text(model, tokenizer, chat_prompt, max_tokens, temp, top_k)
