"""Command-line predictions; diagnostics go to stderr, results to stdout/CSV."""

import argparse
import csv
import json
from pathlib import Path
import sys


def build_parser():
    parser = argparse.ArgumentParser(
        prog="prostab",
        description="Predict independent single substitutions with ProStab.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("predict", "predict-batch"):
        command = commands.add_parser(name)
        command.add_argument(
            "--model-dir",
            default="model_weight",
            help="Directory containing .hydra/config.yaml and checkpoints/best.ckpt",
        )
        command.add_argument("--mpnn-weights", help="Path to v_48_020.pt")
        command.add_argument(
            "--esm-weights",
            help="Local esm2_t33_650M_UR50D.pt; otherwise use Torch Hub",
        )
        command.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
        if name == "predict":
            command.add_argument("--pdb", required=True)
            command.add_argument("--chain", required=True)
            command.add_argument("--mutation", required=True)
        else:
            command.add_argument(
                "--input", required=True, help="CSV with pdb_path,chain,mutation"
            )
            command.add_argument(
                "--output",
                required=True,
                help="New results CSV (never overwrites existing files)",
            )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        rows = None
        if args.command == "predict-batch":
            source = Path(args.input).expanduser().resolve()
            output = Path(args.output).expanduser()
            if output.exists():
                raise ValueError("Output already exists: {}".format(output))
            with source.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if not {"pdb_path", "chain", "mutation"}.issubset(
                    reader.fieldnames or []
                ):
                    raise ValueError(
                        "Input CSV requires pdb_path,chain,mutation columns."
                    )
                rows = list(reader)
            if not rows:
                raise ValueError("Input CSV contains no predictions.")
        from prostab.inference import Predictor

        predictor = Predictor(
            args.model_dir,
            args.device,
            mpnn_weights=args.mpnn_weights,
            esm_weights=args.esm_weights,
        )
        if args.command == "predict":
            value = predictor.predict(args.pdb, args.chain, args.mutation)
            print(
                json.dumps(
                    dict(
                        pdb_path=args.pdb,
                        chain=args.chain,
                        mutation=args.mutation,
                        ddg=value,
                        status="success",
                        error="",
                    )
                )
            )
            return 0
        failed = False
        fields = ["pdb_path", "chain", "mutation", "ddg", "status", "error"]
        with output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                original = {key: row.get(key) or "" for key in fields[:3]}
                if not all(original.values()) or None in row:
                    result = dict(
                        original,
                        ddg=None,
                        status="error",
                        error="Malformed CSV row or missing required value.",
                    )
                else:
                    path = Path(original["pdb_path"]).expanduser()
                    if not path.is_absolute():
                        path = source.parent / path
                    result = predictor._predict_record(
                        path, original["chain"], original["mutation"]
                    )
                    result.update(original)
                writer.writerow(result)
                handle.flush()
                failed |= result["status"] != "success"
        return 1 if failed else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print("prostab: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
