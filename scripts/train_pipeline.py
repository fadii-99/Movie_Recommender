from __future__ import annotations
import argparse
from recommender.train import train_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to client's Excel ratings file") 
    ap.add_argument("--models_dir", default="models", help="Directory to save models") 
    ap.add_argument("--n_components", type=int, default=20)
    args = ap.parse_args()
    meta = train_all(args.excel, args.models_dir, n_components=args.n_components)
    print("Training complete:\n", meta)
