import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio


def encode_dataset(args):
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoints")
    cpc, kmeans = torch.hub.load("bshall/cpc", "cpc")
    cpc = cpc.cuda()

    print(f"Encoding dataset at {in_dir}")
    for in_path in tqdm(sorted(list(in_dir.rglob("*.wav")))):
        wav, sr = torchaudio.load(in_path)
        assert sr == 16000

        wav = wav.unsqueeze(0).cuda()
        x = cpc.encode(wav)

        relative_path = in_path.relative_to(in_dir)
        out_path = out_dir / relative_path.with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(out_path.with_suffix(".npy"), x.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess an audio dataset.")
    parser.add_argument("in_dir", help="Path to the directory to encode.")
    parser.add_argument("out_dir", help="Path to the output directory.")
    args = parser.parse_args()
    encode_dataset(args)
