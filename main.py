import argparse
from pathlib import Path
import nsnet2.enhance_onnx as enhance
from hmm import transcribe_hmm
from w import transcribe_w
import soundfile as sf
import librosa
import os


def main(args):
    # check input path
    input_path = Path(args.i).resolve()
    assert input_path.exists(), "File not found!"

    # check language
    assert args.lan in ("rus", "eng"), "Wrong Language!"

    # check model
    assert args.model in ("w", "hmm"), "Wrong model!"
    if args.model == "hmm" and args.lan == "eng":
        assert False, "Can`t use hmm for english!"

    input_sig, sr = sf.read(args.i)

    # check sample
    if sr == 16000 and args.resample is True:
        assert False, "Input audio`s samplerate is already 16000!"
    if sr != 16000 and args.resample is True:
        input_sig = librosa.resample(y=input_sig, orig_sr=sr, target_sr=16000)
        sr = 16000
    elif sr != 16000 and args.resample is False and args.den is True:
        assert False, "Either change input file, or turn on resampling, or turn off denoise!"
    elif sr != 16000 and args.resample is False and args.model == "hmm":
        assert False, "Either change input file, or turn on resampling, or use another model!"

    script_dir = os.path.dirname(os.path.realpath(__file__))

    # denoise
    if args.den is True:
        enhancer = enhance.NSnet2Enhancer(fs=sr)
        out_sig = enhancer(input_sig, sr)
        p = Path(script_dir, 'denoised.wav')
        sf.write(p, out_sig, sr)
        input_path = p

    # transcribe
    result = ""
    if args.lan == "eng":
        result = transcribe_w(input_path, script_dir)
    elif args.lan == "rus":
        if args.model == "w":
            result = transcribe_w(input_path, script_dir)
        elif args.model == "hmm":
            result = transcribe_hmm(input_path, script_dir)

    # result
    output = open('res.txt', 'w')
    output.write(result)
    output.close()
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", type=str, help="Path to speech wav file", required=True)
    parser.add_argument("-lan", type=str, help="Language", choices=["rus", "eng"], required=True)
    parser.add_argument("-model", type=str, help="Model for transcribing. Available models:\n"
                                                 "whisper - for english and russian;\n"
                                                 "hmm - for russian", choices=["w", "hmm"], default="w")
    parser.add_argument("-den", type=bool, help="Enable/Disable denoiser", choices=[True, False], required=True)
    parser.add_argument("-resample", type=bool, help="If you want to use hmm or denoise, "
                                                     "input audio must have 16kHz samplerate. "
                                                     "Use this flag, if your audio has different samplerate. "
                                                     "This may degrade the recognition quality."
                        , choices=[True, False], default=False)

    args = parser.parse_args()

    main(args)
