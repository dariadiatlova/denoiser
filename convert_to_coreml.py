import argparse

import coremltools as ct
import torch

from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.mil import Builder as mb

from denoiser.audio import Audioset
from denoiser.demucs import Demucs
from denoiser.enhance import write
from torch.utils.data import DataLoader


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-p', '--audio_file_path',
                        help='Path to the wav file to evaluate.',
                        type=str)
    parser.add_argument('-w', '--model_path',
                        help='Path to the .ckpt file to initialize model weights.',
                        type=str)
    parser.add_argument('-o', '--output_wav_path',
                        help='Path of the generated audio, default is data/generated_samples/'
                             '<initial_audio_name>_generated.wav',
                        type=str)
    parser.add_argument('-cp', '--core_ml_model_path',
                        help='Path to the Core-Ml model with <.mlmodel> extension.',
                        type=str)
    parser.add_argument('-sr', '--sample_rate',
                        help='Sample rate of the input audio file. Default is 48000.',
                        type=str,
                        default=16000)
    parser.add_argument('-d', '--device',
                        help='cuda or cpu',
                        type=str,
                        default="cpu")
    parser.add_argument('-dr', '--dry',
                        help='Percentage of mixing noise with clean',
                        type=float,
                        default=0)
    parser.add_argument('-st', '--streaming',
                        help='If use streaming evaluation',
                        type=float,
                        default=False)


def main(params):
    checkpoint = torch.load(params.model_path)
    model = Demucs()
    model.load_state_dict(checkpoint)
    model.eval()

    dset = Audioset([(params.audio_file_path, None)], with_path=True, sample_rate=params.sample_rate,
                    channels=model.chin, convert=True)
    loader = DataLoader(dset, shuffle=True, batch_size=1)

    noisy_signals, filenames = [s for s in loader][0]
    noisy_signals = noisy_signals.to(params.device)

    traced_model = torch.jit.trace(model, noisy_signals)
    out = traced_model(noisy_signals)
    write(out.squeeze(1), params.output_wav_path, sr=params.sample_rate)

    @register_torch_op
    def glu(context, node):
        inputs = _get_inputs(context, node, expected=2)[0]
        x1 = mb.slice_by_index(x=inputs, begin=[0, 0, 0],
                               end=[inputs.shape[0], inputs.shape[1] // 2, inputs.shape[2]])
        x2 = mb.slice_by_index(x=inputs, begin=[0, inputs.shape[1] // 2, 0],
                               end=[inputs.shape[0], inputs.shape[1], inputs.shape[2]])
        x2 = mb.sigmoid(x=x2, name=node.name + "_sigmoid")
        x = mb.mul(x=x1, y=x2)
        context.add(x, torch_name=node.name)

    audio_input = ct.TensorType(shape=noisy_signals.shape)
    coremlmodel = ct.convert(traced_model, inputs=[audio_input])
    coremlmodel.save(params.core_ml_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    params = parser.parse_args()
    main(params)
