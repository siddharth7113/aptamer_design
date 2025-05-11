import argparse
import yaml
from aptamer_design.pipeline import run_inference_pipeline

def main():
    parser = argparse.ArgumentParser(description="Aptamer Binding Affinity Predictor")

    parser.add_argument("--config", type=str, help="Path to YAML config file (overrides all args below)")

    # Individual overrides
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--data", type=str, required=False)
    parser.add_argument("--filetype", type=str, default="csv")
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--seq_col", type=str, default="sequence")
    parser.add_argument("--binding_col", type=str, default="binding")
    parser.add_argument("--id_col", type=str, default="id")

    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=100)

    parser.add_argument("--eval", action="store_true", help="Compute evaluation metrics")

    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        # Fallback: construct config from CLI
        # Only include relevant args based on model
        if args.model == "mlp":
            model_kwargs = {
                "k": args.k,
                "hidden_dim": args.hidden_dim
            }
        elif args.model == "cnn":
            model_kwargs = {
                "max_len": args.max_len
            }
        else:
            model_kwargs = {}

        config = {
            "model_type": args.model,
            "weights_path": args.weights,
            "data_path": args.data,
            "filetype": args.filetype,
            "seq_col": args.seq_col,
            "binding_col": args.binding_col,
            "id_col": args.id_col,
            "output_path": args.output,
            "compute_metrics": args.eval,
            "model_kwargs": model_kwargs
        }

    run_inference_pipeline(config)

if __name__ == "__main__":
    main()
