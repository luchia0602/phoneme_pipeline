import json
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    snrs = [20, 15, 10, 5, 0, -5, -10, -15, -20, -25]
    metrics_dir = Path("data/metrics")
    plt.figure(figsize=(10, 6))
    for lang_dir in metrics_dir.iterdir():
        if not lang_dir.is_dir():
            continue        
        lang = lang_dir.name
        noisy_pers = []
        for snr in snrs:
            metric_file = lang_dir / f"noisy_{snr}_metrics.json"
            if metric_file.exists():
                with open(metric_file) as f:
                    noisy_pers.append(json.load(f)["PER"])
            else:
                noisy_pers.append(None)
        plt.plot(snrs, noisy_pers, marker='o', linewidth=2, label=f'{lang.upper()} PER')

    plt.gca().invert_xaxis()
    plt.xlabel("Signal to noise ratio (lower=noisier)")
    plt.ylabel("PER (lower=better)")
    plt.title("Model robustness to noise across languages")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    out_dir = Path("data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "cross_lingual.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')    
    print("Success!!")

if __name__ == "__main__":
    main()