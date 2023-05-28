from pocketsphinx import Decoder, Segmenter
from pathlib import Path
import wave


def transcribe_hmm(input_path, script_dir):
    path_to_hmm = str(Path(script_dir, "models", "rus", "acoustic"))
    path_to_lm = str(Path(script_dir, "models", "rus", "ru.lm.bin"))
    path_to_dic = str(Path(script_dir, "models", "rus", "ru.dic"))

    result = ""
    with wave.open(input_path.parts[-1], "rb") as w:
        decoder = Decoder(hmm=path_to_hmm,
                          lm=path_to_lm,
                          dict=path_to_dic,
                          samprate=16000)

        segment = Segmenter(sample_rate=16000)

        for seg in segment.segment(w.getfp()):
            decoder.start_utt()
            decoder.process_raw(seg.pcm, full_utt=True)
            decoder.end_utt()
            result += str(decoder.hyp().hypstr)

    return result
