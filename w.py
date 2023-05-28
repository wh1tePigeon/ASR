import whisper
import torch
from pathlib import Path


def transcribe_w(path, script_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    path_to_w = str(Path(script_dir, "models", "eng"))
    model = whisper.load_model(name="medium", download_root=path_to_w)
    model.to(device)

    result = model.transcribe(str(path))

    return result['text']
