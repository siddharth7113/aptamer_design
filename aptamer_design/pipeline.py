import pandas as pd
from aptamer_design.data import AptamerDataset
from aptamer_design.models import load_model
from aptamer_design.evaluation import compute_metrics

def run_inference_pipeline(
    config: dict,
    return_outputs: bool = False
):
    """
    Unified function to run aptamer inference and optional evaluation.

    Args:
        config (dict): Contains keys:
            - model_type (str)
            - weights_path (str or None)
            - model_kwargs (dict)
            - data_path (str)
            - filetype (str)
            - seq_col (str)
            - binding_col (str or None)
            - id_col (str or None)
            - output_path (str or None)
            - compute_metrics (bool)
        return_outputs (bool): If True, return predictions in memory.

    Returns:
        Optional: list of dicts if return_outputs=True
    """
    # 1. Load model
    model = load_model(
        model_type=config["model_type"],
        weights_path=config.get("weights_path"),
        **config.get("model_kwargs", {})
    )

    # 2. Load dataset
    dataset = AptamerDataset.from_table(
        path=config["data_path"],
        filetype=config["filetype"],
        seq_col=config["seq_col"],
        binding_col=config.get("binding_col", "binding"),
        id_col=config.get("id_col", "id")
    )

    # 3. Predict
    sequences = [sample.sequence for sample in dataset]
    preds = model.predict(sequences)

    # 4. Prepare output rows
    rows = []
    for i, sample in enumerate(dataset):
        row = {
            "id": sample.id or f"seq{i}",
            "sequence": sample.sequence,
            "prediction": float(preds[i])
        }
        if sample.binding is not None:
            row["binding"] = sample.binding
        rows.append(row)

    df_out = pd.DataFrame(rows)

    # 5. Evaluate
    if config.get("compute_metrics", True) and dataset.has_labels:
        y_true = df_out["binding"].values
        y_pred = df_out["prediction"].values
        metrics = compute_metrics(y_true, y_pred)
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k:10}: {v:.4f}")
    else:
        metrics = {}

    # 6. Save to disk
    if config.get("output_path"):
        df_out.to_csv(config["output_path"], index=False)

    # 7. Return if needed
    return (df_out, metrics) if return_outputs else None
