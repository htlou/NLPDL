# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Dataset class for preference training."""

from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch

from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset, RawDataset
from safe_rlhf.datasets.preference import PreferenceCollator, PreferenceSample, PreferenceBatch
from safe_rlhf.datasets.utils import format_prompt, right_padding
import json

__all__ = [
    'NLPDLPreferenceDataset',
    'NLPDLSupervisedDataset',
]

IGNORE_INDEX = -100

class NLPDLPreferenceDataset(RawDataset):
    NAME: str = 'nlpdl-preference'

    def __init__(self, path) -> None:  # noqa: ANN001
        self.path = path
        with open(self.path, encoding='utf-8') as f:
            self.data = json.load(f)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['prompt']
        chosen = data['chosen']
        rejected = data['rejected']
        return RawSample(input=input, answer=chosen, other_answer=rejected, better=True)

    def __len__(self) -> int:
        return len(self.data)

class NLPDLSupervisedDataset(RawDataset):
    NAME: str = 'nlpdl-supervised'

    def __init__(self, path) -> None:  # noqa: ANN001
        self.path = path
        with open(self.path, encoding='utf-8') as f:
            self.data = json.load(f)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['prompt']
        answer = data['response']
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)
