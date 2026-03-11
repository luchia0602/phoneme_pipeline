import json
import os
import argparse
from pathlib import Path

# Tell phonemizer exactly where the eSpeak NG library is
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def main(input_manifest, out_manifest):
    in_path = Path(input_manifest)
    out_path = Path(out_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix('.tmp')

    # Load the specific model required by the lab
    model_id = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()

    with open(in_path, 'r', encoding='utf-8') as f_in, \
         open(temp_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            entry = json.loads(line)
            wav_path = entry['wav_path']
            
            speech, sr = sf.read(wav_path)
            
            # --- NEW: Automatically resample to 16000Hz if needed ---
            if sr != 16000:
                speech_tensor = torch.tensor(speech).float()
                # If the audio is stereo (2 channels), average it to mono
                if speech_tensor.ndim > 1:
                    speech_tensor = speech_tensor.mean(dim=1)
                # Resample the tensor
                speech_tensor = torchaudio.functional.resample(speech_tensor, orig_freq=sr, new_freq=16000)
                speech = speech_tensor.numpy()
                sr = 16000
            # --------------------------------------------------------
            
            input_values = processor(speech, sampling_rate=sr, return_tensors="pt").input_values
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            
            entry['pred_phon'] = transcription
            f_out.write(json.dumps(entry) + '\n')
            
    os.replace(temp_path, out_path)
    print(f"Prediction manifest created at: {out_path}. Success!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.input, args.out)