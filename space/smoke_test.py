from __future__ import annotations

import argparse

import app as spa


def run_lightweight_checks() -> None:
    triplet_id = spa._all_triplet_ids()[0]
    sample_name = spa.TRIPLETS_BY_ID[triplet_id]["sample_images"][0]
    sample_path = spa._resolve_valid_sample_image_path(triplet_id, sample_name)
    if not sample_path.exists():
        raise RuntimeError(f"Sample image missing: {sample_path}")

    gt_camera = spa._compute_gt_label(sample_path, triplet_id=triplet_id, perspective="camera")
    gt_human = spa._compute_gt_label(sample_path, triplet_id=triplet_id, perspective="human")
    if not isinstance(gt_camera, str) or not isinstance(gt_human, str):
        raise RuntimeError("GT label computation failed.")

    print(f"[OK] manifest/triplet/sample checks passed ({triplet_id}, {sample_name})")
    print(f"[OK] GT labels camera={gt_camera}, human={gt_human}")


def run_inference_check() -> None:
    triplet_id = spa._all_triplet_ids()[0]
    sample_name = spa.TRIPLETS_BY_ID[triplet_id]["sample_images"][0]
    model_key = spa._initial_model_key()

    pred_html, labels_html, overlay, heatmap, _, _, _, _, _, run_info_html, _, _ = spa._infer_core(
        perspective="camera",
        model_key=model_key,
        triplet_id=triplet_id,
        source_mode="Sample gallery",
        sample_image_name=sample_name,
        uploaded_image=None,
        map_name="mean",
        alpha=0.25,
        compare_enabled=False,
        compare_map_name="q1",
        sharpen=False,
        clip_percentile=98.0,
        gamma=0.65,
    )

    if not pred_html or not labels_html or not run_info_html:
        raise RuntimeError("Inference HTML outputs are empty.")
    if getattr(overlay, "shape", None) is None:
        raise RuntimeError("Overlay output type is unexpected.")
    if getattr(heatmap, "shape", None) is None:
        raise RuntimeError("Heatmap output type is unexpected.")

    print(f"[OK] inference check passed with model={model_key}, triplet={triplet_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SpaRRTA local smoke checks")
    parser.add_argument(
        "--run-infer",
        action="store_true",
        help="Run one full model inference (downloads/loads timm backbone if needed).",
    )
    args = parser.parse_args()

    run_lightweight_checks()
    if args.run_infer:
        run_inference_check()


if __name__ == "__main__":
    main()
