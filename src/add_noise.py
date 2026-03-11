import json
import os
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import hashlib

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal.shape)
    return signal + noise

def add_noise_to_file(input_wav: str, output_wav: str, snr_db: float, seed: int = None) -> None:
    signal, sr = sf.read(input_wav)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)
    sf.write(output_wav, noisy_signal, sr)

def main(input_manifest, out_dir, snr_str, seed):
    in_path = Path(input_manifest)
    out_dir_path = Path(out_dir)
    snr_float = float(snr_str)
    wav_out_dir = out_dir_path / 'wav'
    wav_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir_path / f'noisy_{snr_str}db.jsonl'
    temp_path = manifest_out.with_suffix('.tmp')

    with open(in_path, 'r', encoding='utf-8') as f_in, \
             open(temp_path, 'w', encoding='utf-8') as f_out:
            
        for line in f_in:
            entry = json.loads(line)
            clean_wav = entry['wav_path']
            stem = Path(clean_wav).stem
            noisy_wav_name = f"{stem}_snr{snr_str}.wav"
            noisy_wav_path = wav_out_dir / noisy_wav_name
            add_noise_to_file(clean_wav, str(noisy_wav_path), snr_float, seed)
            entry['wav_path'] = noisy_wav_path.as_posix()
            entry['snr_db'] = snr_float
            entry['audio_md5'] = get_md5(noisy_wav_path)            
            f_out.write(json.dumps(entry) + '\n')
            
    os.replace(temp_path, manifest_out)
    print(f"Manifest file created at: {manifest_out}. Success!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--snr", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.input, args.out_dir, args.snr, args.seed)