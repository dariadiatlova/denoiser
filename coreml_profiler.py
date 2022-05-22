import soundfile as sf
import librosa
import time
import coremltools as ct
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--audio_filepath", type=str, help="path to the audio file to enhance")
parser.add_argument("--model_weights", type=str, help="path to the model weights", default="weights/denoiser.mlmodel")


def main(args):
    start_time = time.time()
    interpreter = ct.models.MLModel(args.model_weights)
    print('Model initialization took [s]:')
    print(time.time() - start_time)

    audio, fs = librosa.load(args.audio_filepath, sr=16000)
    start_time = time.time()
    enhanced = interpreter.predict({"noisy_wav_signal": np.reshape(audio, [1, 1, 240000])})["var_748"]
    print('Model run took [s]:')
    print(time.time() - start_time)

    sf.write(f"{args.audio_filepath[:-4]}_enhanced.wav", np.reshape(enhanced, 240000), fs)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
