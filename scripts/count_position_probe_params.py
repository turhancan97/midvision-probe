from __future__ import annotations

import argparse


def bool_arg(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def count_classification_head(feat_dim: int, num_classes: int, use_layernorm: bool) -> int:
    norm = (2 * feat_dim) if use_layernorm else 0
    classifier = feat_dim * num_classes + num_classes
    return norm + classifier


def count_efficient_probing(
    feat_dim: int,
    num_classes: int,
    use_layernorm: bool,
    num_queries: int,
    d_out: int,
    qkv_bias: bool,
) -> int:
    if feat_dim % d_out != 0:
        raise ValueError(f"feat_dim ({feat_dim}) must be divisible by d_out ({d_out}).")
    reduced_dim = feat_dim // d_out
    norm = (2 * reduced_dim) if use_layernorm else 0
    v = feat_dim * reduced_dim + (reduced_dim if qkv_bias else 0)
    cls_token = num_queries * feat_dim
    classifier = reduced_dim * num_classes + num_classes
    return norm + v + cls_token + classifier


def count_abmilp_head(
    feat_dim: int,
    num_classes: int,
    use_layernorm: bool,
    depth: int,
    self_attention_apply_to: str,
) -> int:
    if depth < 1:
        raise ValueError("abmil depth must be >= 1.")
    # attention_predictor: (depth-1) x Linear(D,D) + final Linear(D,1)
    predictor = (depth - 1) * (feat_dim * feat_dim + feat_dim) + (feat_dim + 1)
    # Attention(dim=D, num_heads=1, qkv_bias=False, qk_norm=False):
    # qkv weight: D x 3D ; proj weight: D x D ; proj bias: D
    self_attn = 0
    if self_attention_apply_to != "none":
        self_attn = 4 * feat_dim * feat_dim + feat_dim
    norm = (2 * feat_dim) if use_layernorm else 0
    classifier = feat_dim * num_classes + num_classes
    return predictor + self_attn + norm + classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count trainable parameters for position probe heads."
    )
    parser.add_argument("--feat-dim", type=int, default=768, help="Backbone feature dimension.")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of output classes.")
    parser.add_argument(
        "--use-layernorm",
        type=bool_arg,
        default=True,
        help="Enable LayerNorm in heads where supported (true/false).",
    )

    parser.add_argument("--efficient-num-heads", type=int, default=1)
    parser.add_argument("--efficient-num-queries", type=int, default=4)
    parser.add_argument("--efficient-d-out", type=int, default=8)
    parser.add_argument("--efficient-qkv-bias", type=bool_arg, default=False)

    parser.add_argument(
        "--abmil-self-attention-apply-to",
        type=str,
        default="none",
        choices=["none", "map", "both"],
    )
    parser.add_argument("--abmil-activation", type=str, default="relu", choices=["relu", "tanh"])
    parser.add_argument("--abmil-depth", type=int, default=1)
    parser.add_argument("--abmil-cond", type=str, default="none", choices=["none", "pe"])
    parser.add_argument("--abmil-content", type=str, default="all", choices=["all", "patch"])
    parser.add_argument(
        "--abmil-num-patches",
        type=int,
        default=196,
        help="Only used when --abmil-cond=pe.",
    )

    args = parser.parse_args()

    rows = [
        (
            "ClassificationHead",
            count_classification_head(
                feat_dim=args.feat_dim,
                num_classes=args.num_classes,
                use_layernorm=args.use_layernorm,
            ),
        ),
        (
            "EfficientProbing",
            count_efficient_probing(
                feat_dim=args.feat_dim,
                num_classes=args.num_classes,
                use_layernorm=args.use_layernorm,
                num_queries=args.efficient_num_queries,
                d_out=args.efficient_d_out,
                qkv_bias=args.efficient_qkv_bias,
            ),
        ),
        (
            "ABMILPHead",
            count_abmilp_head(
                feat_dim=args.feat_dim,
                num_classes=args.num_classes,
                use_layernorm=args.use_layernorm,
                depth=args.abmil_depth,
                self_attention_apply_to=args.abmil_self_attention_apply_to,
            ),
        ),
    ]

    print("Probe parameter counts (trainable only)")
    print(f"feat_dim={args.feat_dim}, num_classes={args.num_classes}, use_layernorm={args.use_layernorm}")
    print("-" * 54)
    for name, count in rows:
        print(f"{name:<22} {count:>12,d}")


if __name__ == "__main__":
    main()
