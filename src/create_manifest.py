import json
import os
import hashlib
import subprocess
from pathlib import Path
import argparse

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_phonemes(text, lang):
    espeak_exe = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    result = subprocess.run(
            [espeak_exe, '-q', '-v', lang, '--ipa', text],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
    return result.stdout.strip()

def main(raw_dir, output_manifest, lang):
    out_path = Path(output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix('.tmp')
    wav_dir = Path(raw_dir) / lang / 'wav'
    transcript_file = Path(raw_dir) / lang / 'transcripts.json'
    with open(transcript_file, 'r', encoding='utf-8') as f:
            transcripts = json.load(f)

    with open(temp_path, 'w', encoding='utf-8') as f:
        for wav_file in wav_dir.glob('*.wav'):
            stem = wav_file.stem
            ref_text = transcripts.get(stem, "Unknown text")
            entry = {
                "utt_id": f"{lang}_{stem}",
                "lang": lang,
                "wav_path": wav_file.as_posix(), 
                "ref_text": ref_text,
                "ref_phon": get_phonemes(ref_text, lang),
                "audio_md5": get_md5(wav_file)
            }
            f.write(json.dumps(entry) + '\n')
            
    os.replace(temp_path, out_path)
    print(f"Manifest file created at: {out_path}. Success!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="Path to raw data")
    parser.add_argument("--out", required=True, help="Path to output manifest")
    parser.add_argument("--lang", required=True, help="Language code")
    args = parser.parse_args()
    main(args.raw_dir, args.out, args.lang)