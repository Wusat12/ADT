
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import lfilter
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "io"))

from models.trainable_pcen import TrainablePCEN
from load import add_delta_features


def fast_fixed_pcen(mel, parameters, eps=1e-6):
    """Fixed PCEN over a complete [time, 84, 1] recording."""

    if mel.ndim != 3 or mel.shape[1:] != (84, 1):
        raise ValueError(f"Unexpected Mel shape: {tuple(mel.shape)}")

    if mel.shape[0] == 0 or mel.shape[0] % 400:
        raise ValueError("Recording length must be divisible by 400")

    x = (
        mel.squeeze(-1)
        .transpose(0, 1)
        .contiguous()
        .numpy()
        .astype(np.float32, copy=False)
    )

    x = np.maximum(x, 0)

    s = parameters["s"]
    alpha = parameters["alpha"]
    delta = parameters["delta"]
    r = parameters["r"]

    # M[:, 0] = x[:, 0]; identical EMA initial condition.
    zi = (1.0 - s) * x[:, :1]

    smooth = lfilter(
        [s],
        [1.0, -(1.0 - s)],
        x,
        axis=1,
        zi=zi,
    )[0].astype(np.float32)

    # Same PCEN equation as TrainablePCEN.
    x_tensor = torch.from_numpy(x)
    smooth_tensor = torch.from_numpy(smooth)

    pcen = (
        x_tensor / (eps + smooth_tensor).pow(alpha)
        + delta
    ).pow(r) - delta**r

    features = add_delta_features(pcen)
    features = features.T.contiguous()

    return features.reshape(-1, 400, 168, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data" / "ENST_IDMT_Polyphonic",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    pcen_file = (
        checkpoint if checkpoint.name == "pcen.pt"
        else checkpoint / "pcen.pt"
    )

    pcen = TrainablePCEN()
    pcen.load_state_dict(
        torch.load(
            pcen_file,
            map_location="cpu",
            weights_only=True,
        )
    )

    pcen.eval()

    parameters = {
        name: float(value.detach())
        for name, value in pcen.get_parameters().items()
    }

    print("Learned PCEN parameters:", parameters, flush=True)

    root = args.dataset_root.resolve()
    splits = ("train", "validation", "test")

    jobs = []
    for split in splits:
        files = sorted(
            (root / "mel_recordings" / split).glob("*/*.pt")
        )

        if not files:
            raise RuntimeError(f"No recordings for {split}")

        output = (
            root /
            f"enst_idmt_polyphonic_optimized_pcen_{split}.pt"
        )

        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing dataset: {output}"
            )

        jobs.append((split, files, output))

    for split, files, output in jobs:
        all_features = []
        all_labels = []

        print(
            f"\nStarting {split}: {len(files)} recordings",
            flush=True,
        )

        for index, file in enumerate(files, 1):
            record = torch.load(
                file,
                map_location="cpu",
                weights_only=True,
            )

            mel = record["mel"].float()
            labels = record["labels"].float()

            if labels.shape != (mel.shape[0], 3):
                raise ValueError(
                    f"Label shape mismatch: {file}"
                )

            with torch.no_grad():
                features = fast_fixed_pcen(
                    mel, parameters, eps=pcen.eps
                )
                label_chunks = labels.reshape(-1, 400, 3)

            all_features.append(features)
            all_labels.append(label_chunks)

            if index % 10 == 0 or index == len(files):
                print(
                    f"{split}: {index}/{len(files)} recordings",
                    flush=True,
                )

        dataset = TensorDataset(
            torch.cat(all_features, dim=0),
            torch.cat(all_labels, dim=0),
        )

        torch.save(dataset, output)

        print(
            f"✅ Saved {output.name}: "
            f"{len(dataset)} chunks",
            flush=True,
        )

        del all_features, all_labels, dataset

    metadata = {
        "frontend": "optimized_pcen",
        "source_checkpoint": str(pcen_file),
        "parameters": parameters,
        "note": "Parameters from interrupted learnable-PCEN training",
    }

    (root / "optimized_pcen_parameters.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n✅ All optimized fixed-PCEN datasets generated.")


if __name__ == "__main__":
    main()
