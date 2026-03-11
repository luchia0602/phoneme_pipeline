import json
import argparse
import soundfile as sf
from datasets import load_dataset
from pathlib import Path

def main(lang_config, espeak_lang, out_dir, num_samples):
    wav_dir = Path(out_dir) / espeak_lang / 'wav'
    wav_dir.mkdir(parents=True, exist_ok=True)
    print(f"Number of examples: {num_samples} from ({lang_config})...")
    dataset = load_dataset("facebook/multilingual_librispeech", lang_config, split="dev", streaming=True)
    transcripts = {}
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        text = sample["text"]
        file_id = sample["id"]
        wav_path = wav_dir / f"{file_id}.wav"
        sf.write(wav_path, audio_array, sr)
        transcripts[file_id] = text
    transcripts_path = Path(out_dir) / espeak_lang / "transcripts.json"
    with open(transcripts_path, 'w', encoding='utf-8') as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    print(f"Saved {num_samples} files to {wav_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_config", required=True, help="HF Dataset language config (e.g., 'english', 'german')")
    parser.add_argument("--espeak_lang", required=True, help="espeak-ng language code (e.g., 'en', 'de')")
    parser.add_argument("--out_dir", required=True, help="Base output directory for raw data")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of audio files to download")
    args = parser.parse_args()
    main(args.lang_config, args.espeak_lang, args.out_dir, args.num_samples)