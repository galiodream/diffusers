import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# -------------------------------
# 1) 加载 tokenizer / model
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)

# -------------------------------
# 2) 辅助函数：把模块写成更细的可读结构
# -------------------------------
def module_brief(module: nn.Module) -> str:
    """
    针对常见层给出更细的描述。
    """
    cls_name = module.__class__.__name__

    if isinstance(module, nn.Linear):
        return (
            f"{cls_name}("
            f"in_features={module.in_features}, "
            f"out_features={module.out_features}, "
            f"bias={module.bias is not None})"
        )

    if isinstance(module, nn.Embedding):
        return (
            f"{cls_name}("
            f"num_embeddings={module.num_embeddings}, "
            f"embedding_dim={module.embedding_dim})"
        )

    # 常见 norm 层：有些自定义 RMSNorm 没有 normalized_shape，做兜底
    if hasattr(module, "weight") and isinstance(getattr(module, "weight", None), torch.Tensor):
        wshape = tuple(module.weight.shape)
        if "Norm" in cls_name:
            return f"{cls_name}(weight_shape={wshape})"

    # 激活函数
    if isinstance(module, (nn.SiLU, nn.GELU, nn.ReLU, nn.Tanh, nn.Sigmoid)):
        return f"{cls_name}()"

    # 其他模块：至少给类名
    return f"{cls_name}()"


def dump_module_tree(module: nn.Module, name: str = "model", indent: int = 0) -> str:
    """
    递归展开模块树，不压缩 ModuleList，真正把每一层展开。
    """
    prefix = "  " * indent
    lines = []

    children = list(module.named_children())
    if not children:
        lines.append(f"{prefix}{name}: {module_brief(module)}")
        return "\n".join(lines)

    lines.append(f"{prefix}{name}: {module.__class__.__name__}")

    # 对 ModuleList / Sequential，逐项展开
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        for idx, child in enumerate(module):
            lines.append(dump_module_tree(child, name=f"[{idx}]", indent=indent + 1))
        return "\n".join(lines)

    # 普通模块按 child name 展开
    for child_name, child in children:
        lines.append(dump_module_tree(child, name=child_name, indent=indent + 1))

    return "\n".join(lines)


def dump_named_parameters(module: nn.Module) -> str:
    """
    保存所有参数名、shape、是否可训练。
    """
    lines = []
    for name, param in module.named_parameters():
        lines.append(
            f"{name:<80} shape={tuple(param.shape)!s:<20} requires_grad={param.requires_grad}"
        )
    return "\n".join(lines)


def dump_config_summary(model) -> str:
    """
    保存配置里的关键信息，便于对照实际结构理解。
    """
    cfg = model.config
    fields = [
        "model_type",
        "architectures",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "hidden_act",
        "rms_norm_eps",
        "rope_theta",
        "torch_dtype",
    ]

    lines = ["[Config Summary]"]
    for k in fields:
        v = getattr(cfg, k, "N/A")
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def dump_qwen25_block_summary(model) -> str:
    """
    额外给出一个“按 Qwen2.5 结构理解”的摘要，
    帮你把 q/k/v、head_dim、mlp 等关键信息写清楚。
    """
    lines = ["[Qwen2.5 Structure Summary]"]

    cfg = model.config
    hidden_size = getattr(cfg, "hidden_size", None)
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_kv_heads = getattr(cfg, "num_key_value_heads", None)
    intermediate_size = getattr(cfg, "intermediate_size", None)
    num_layers = getattr(cfg, "num_hidden_layers", None)
    vocab_size = getattr(cfg, "vocab_size", None)

    head_dim = None
    if hidden_size is not None and num_heads is not None and num_heads != 0:
        head_dim = hidden_size // num_heads

    lines.append(f"vocab_size         = {vocab_size}")
    lines.append(f"hidden_size        = {hidden_size}")
    lines.append(f"intermediate_size  = {intermediate_size}")
    lines.append(f"num_hidden_layers  = {num_layers}")
    lines.append(f"num_attention_heads= {num_heads}")
    lines.append(f"num_key_value_heads= {num_kv_heads}")
    lines.append(f"head_dim           = {head_dim}")
    lines.append("")
    lines.append("Overall:")
    lines.append(f"  input_ids [B, T]")
    lines.append(f"  -> embed_tokens")
    lines.append(f"  -> hidden_states [B, T, {hidden_size}]")
    lines.append(f"  -> {num_layers} x DecoderLayer")
    lines.append(f"  -> final norm")
    lines.append(f"  -> lm_head")
    lines.append(f"  -> logits [B, T, {vocab_size}]")
    lines.append("")
    lines.append("Per DecoderLayer:")
    lines.append(f"  x [B, T, {hidden_size}]")
    lines.append(f"  -> input_layernorm")
    lines.append(f"  -> self_attn:")
    lines.append(f"       q_proj: [{hidden_size} -> {hidden_size}]")
    if num_kv_heads is not None and head_dim is not None:
        kv_dim = num_kv_heads * head_dim
        lines.append(f"       k_proj: [{hidden_size} -> {kv_dim}]")
        lines.append(f"       v_proj: [{hidden_size} -> {kv_dim}]")
    lines.append(f"       o_proj: [{hidden_size} -> {hidden_size}]")
    lines.append(f"       rotary_emb on q/k")
    lines.append(f"  -> residual add")
    lines.append(f"  -> post_attention_layernorm")
    lines.append(f"  -> mlp:")
    lines.append(f"       gate_proj: [{hidden_size} -> {intermediate_size}]")
    lines.append(f"       up_proj:   [{hidden_size} -> {intermediate_size}]")
    lines.append(f"       SiLU(gate_proj(x)) * up_proj(x)")
    lines.append(f"       down_proj: [{intermediate_size} -> {hidden_size}]")
    lines.append(f"  -> residual add")

    return "\n".join(lines)


# -------------------------------
# 3) 生成并保存更细的实际结构
# -------------------------------
output_txt = "qwen2.5_model_structure_expanded.txt"
param_txt = "qwen2.5_model_parameters.txt"

structure_text = []
structure_text.append("=" * 100)
structure_text.append("Qwen2.5 Expanded Model Structure")
structure_text.append("=" * 100)
structure_text.append("")
structure_text.append(dump_config_summary(model))
structure_text.append("")
structure_text.append("=" * 100)
structure_text.append(dump_qwen25_block_summary(model))
structure_text.append("")
structure_text.append("=" * 100)
structure_text.append("[Expanded Module Tree]")
structure_text.append("=" * 100)
structure_text.append(dump_module_tree(model, name="model", indent=0))

with open(output_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(structure_text))

with open(param_txt, "w", encoding="utf-8") as f:
    f.write("=" * 100 + "\n")
    f.write("Qwen2.5 Parameter Details\n")
    f.write("=" * 100 + "\n\n")
    f.write(dump_named_parameters(model))

print(f"展开后的模型结构已保存到: {output_txt}")
print(f"参数明细已保存到: {param_txt}")