#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

path=egs/debug/april_100
if [[ ! -e $path ]]; then
    mkdir -p $path
fi

mkdir -p $path/train $path/val $path/test

python3 -m denoiser.audio /home/dadyatlova_1/dataset/main/data_100_hours/noisy_train > $path/train/noisy.json
python3 -m denoiser.audio /home/dadyatlova_1/dataset/main/data_100_hours/noisy_val > $path/val/noisy.json
python3 -m denoiser.audio /home/dadyatlova_1/dataset/main/data_100_hours/noisy_test > $path/test/noisy.json

python3 -m denoiser.audio /home/dadyatlova_1/dataset/main/data_100_hours/clean_train > $path/train/clean.json
python3 -m denoiser.audio /home/dadyatlova_1/dataset/main/data_100_hours/clean_val > $path/val/clean.json
python3 -m denoiser.audio /home/dadyatlova_1/dataset/main/data_100_hours/clean_test > $path/test/clean.json
