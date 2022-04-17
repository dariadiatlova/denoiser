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
python3 -m denoiser.audio /home/dadyatlova_1/dataset/main/data_100_hours/noisy_train > $path/noisy_train.json
python3 -m denoiser.audio /home/dadyatlova_1/dataset/main/data_100_hours/noisy_val > $path/noisy_val.json
python3 -m denoiser.audio /home/dadyatlova_1/dataset/main/data_100_hours/noisy_test > $path/noisy_test.json

python3 -m denoiser.audio /home/dadyatlova_1/dataset/main/data_100_hours/clean_train > $path/clean_train.json
python3 -m denoiser.audio /home/dadyatlova_1/dataset/main/data_100_hours/clean_val > $path/clean_val.json
python3 -m denoiser.audio /home/dadyatlova_1/dataset/main/data_100_hours/clean_test > $path/clean_test.json
