"""NeonBench Model Visualizer — Interactive attention heatmaps.
Run: streamlit run scripts/visualizer.py
"""
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys, importlib
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
    """Best-effort tokenizer path resolution."""
    for prefix in [data_name, "hp"]:
        p = os.path.join(TOK_DIR, f"{prefix}_{tok_name}.json")
        if os.path.exists(p):
            return p
    for f in os.listdir(TOK_DIR):
        if f.endswith(f"_{tok_name}.json"):
            return os.path.join(TOK_DIR, f)
    return None

@st.cache_resource
def load_model(model_name, ckpt_path, tok_name, data_name):
    tok_path = find_tokenizer(tok_name, data_name)
    if tok_path is None:
        return None, None, None
    tokenizer = Tokenizer.from_file(tok_path)

    config = get_config(model_name)
    config['vocab_size'] = tokenizer.get_vocab_size()

    cls_name = model_name.capitalize()
    mod = importlib.import_module(f"models.{model_name}")
    ModelClass = getattr(mod, cls_name)
    model = ModelClass(config)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer, config


def clean_token(t):
    """Make BPE tokens more readable."""
    return t.replace("Ġ", "·").replace("Ċ", "↵").replace("▁", "·")


# ── Attention + MLP capture ──────────────────────────────────────────

def capture_forward(model, input_ids):
    """Run forward pass, capture attention weights and MLP activations."""
    attn_bucket = []
    mlp_bucket = []
    real_sdpa = F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        L, S = q.size(-2), k.size(-2)
        s = 1.0 / (q.size(-1) ** 0.5) if scale is None else scale
        logits = q @ k.transpose(-2, -1) * s
        if is_causal:
            mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=q.device), diagonal=1)
            logits.masked_fill_(mask, float("-inf"))
        if attn_mask is not None:
            logits = logits + attn_mask
        w = torch.softmax(logits, dim=-1)
        attn_bucket.append(w.detach().cpu())
        return w @ v

    # Hook MLP modules to capture their input and output
    hooks = []
    for block in model.blocks:
        if hasattr(block, 'mlp'):
            def make_hook():
                def hook_fn(module, inp, out):
                    mlp_bucket.append({
                        'input': inp[0].detach().cpu(),    # (B, T, d_model)
                        'output': out.detach().cpu(),      # (B, T, d_model)
                    })
                return hook_fn
            hooks.append(block.mlp.register_forward_hook(make_hook()))

    F.scaled_dot_product_attention = spy
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        F.scaled_dot_product_attention = real_sdpa
        for h in hooks:
            h.remove()
    return attn_bucket, mlp_bucket


# ── Plotting ─────────────────────────────────────────────────────────

def plot_single_head(attn_matrix, tokens, layer, head):
    """Interactive plotly heatmap for one attention head."""
    labels = [clean_token(t) for t in tokens]
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
        title=f"Layer {layer} — Head {head}",
        xaxis_title="Key (attended to →)",
        yaxis_title="Query (attending from ↓)",
        yaxis=dict(autorange="reversed"),
        height=max(420, T * 22 + 120),
        margin=dict(l=80, r=20, t=50, b=80),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_all_heads(attn_layer, tokens, layer, n_heads):
    """Small-multiple grid of all heads for one layer."""
    cols = min(n_heads, 4)
    rows = (n_heads + cols - 1) // cols
    labels = [clean_token(t) for t in tokens]

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
    labels = [clean_token(t) for t in tokens]
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
    labels = [clean_token(t) for t in tokens]
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
    # Add row labels on left
    for l in range(n_layers):
        fig.add_annotation(text=f"L{l}", x=-0.02, y=1 - (l + 0.5) / n_layers,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=11, color="gray"))
    return fig


def plot_mlp_activation(mlp_data, tokens, layer):
    """Heatmap of MLP output magnitude per token per dimension (top dims)."""
    out = mlp_data[layer]['output'][0].numpy()  # (T, d_model)
    # Show top-32 most active dimensions (by variance across tokens)
    var = np.var(out, axis=0)
    top_dims = np.argsort(var)[-32:][::-1]
    labels = [clean_token(t) for t in tokens]
    fig = go.Figure(go.Heatmap(
        z=out[:, top_dims].T,
        x=labels,
        y=[f"d{d}" for d in top_dims],
        colorscale="RdBu_r", zmid=0,
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        title=f"MLP Output — Layer {layer} (top 32 dims by variance)",
        xaxis_title="Token", yaxis_title="Dimension",
        height=500, margin=dict(l=60, r=20, t=50, b=80),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def plot_mlp_norm(mlp_data, tokens, n_layers):
    """Heatmap: L2 norm of MLP output per token per layer."""
    labels = [clean_token(t) for t in tokens]
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
    st.error("Could not load model or tokenizer.")
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
""")

# ── Prompt input ──
prompt = st.text_area("✏️ Enter a prompt:",
                      value="Harry looked at the mirror and saw",
                      height=80)

if st.button("🔍 Visualize", type="primary") or "attentions" in st.session_state:
    # Only recompute if prompt changed or first run
    current_key = f"{selected}|{prompt}"
    if st.session_state.get("_vis_key") != current_key:
        encoded = tokenizer.encode(prompt)
        ids = encoded.ids
        toks = encoded.tokens
        if not ids:
            st.error("Tokenizer produced no tokens. Try a different prompt.")
            st.stop()
        if len(ids) > config['block_size']:
            st.warning(f"Truncated to {config['block_size']} tokens.")
            ids = ids[:config['block_size']]
            toks = toks[:config['block_size']]

        input_tensor = torch.tensor([ids])
        attns, mlps = capture_forward(model, input_tensor)

        st.session_state["attentions"] = attns
        st.session_state["mlp_data"] = mlps
        st.session_state["tokens"] = toks
        st.session_state["_vis_key"] = current_key

    attns = st.session_state["attentions"]
    mlps  = st.session_state.get("mlp_data", [])
    toks  = st.session_state["tokens"]

    st.success(f"**{len(toks)} tokens** × {n_layers} layers × {n_heads} heads")
    st.markdown("**Tokens:** " + "  ".join(f"`{clean_token(t)}`" for t in toks))

    # ── Tabs ──
    tabs = st.tabs(["🔎 Single Head", "📊 All Heads", "🧩 Full Grid",
                    "📈 Attention Flow", "🔧 MLP Activations"])

    # --- Single Head ---
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            layer = st.selectbox("Layer", range(n_layers),
                                 format_func=lambda x: f"Layer {x}", key="sh_layer")
        with c2:
            head = st.selectbox("Head", range(n_heads),
                                format_func=lambda x: f"Head {x}", key="sh_head")
        attn_np = attns[layer][0, head].numpy()
        st.plotly_chart(plot_single_head(attn_np, toks, layer, head),
                        use_container_width=True)
        st.plotly_chart(plot_attention_received(attn_np, toks, layer, head),
                        use_container_width=True)

    # --- All Heads (one layer) ---
    with tabs[1]:
        layer_all = st.selectbox("Layer", range(n_layers),
                                 format_func=lambda x: f"Layer {x}", key="ah_layer")
        st.plotly_chart(plot_all_heads(attns[layer_all], toks, layer_all, n_heads),
                        use_container_width=True)

    # --- Full Grid (all layers × all heads) ---
    with tabs[2]:
        st.markdown(f"**{n_layers} layers × {n_heads} heads** — rows = layers, columns = heads")
        st.plotly_chart(plot_full_grid(attns, toks, n_layers, n_heads),
                        use_container_width=True)

    # --- Attention Flow ---
    with tabs[3]:
        st.markdown("**Average attention per layer** (mean across all heads)")
        layer_flow = st.selectbox("Layer", range(n_layers),
                                  format_func=lambda x: f"Layer {x}", key="af_layer")
        avg = attns[layer_flow][0].mean(dim=0).numpy()
        labels = [clean_token(t) for t in toks]
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
