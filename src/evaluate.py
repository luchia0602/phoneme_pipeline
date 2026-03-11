import json
import os
import argparse
from pathlib import Path
from jiwer import cer

def main(input_manifest, out_metrics):
    in_path = Path(input_manifest)
    out_path = Path(out_metrics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    refs = []
    preds = []
    
    with open(in_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            entry = json.loads(line)
            refs.append(entry['ref_phon'])
            preds.append(entry['pred_phon'])
            
    # Calculate PER by utilizing jiwer's Character Error Rate
    error_rate = cer(refs, preds)
    
    metrics = {"PER": error_rate}
    with open(out_path, 'w', encoding='utf-8') as f_out:
        json.dump(metrics, f_out, indent=4)
        
    print(f"Metrics created at: {out_path}. PER: {error_rate}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.input, args.out)