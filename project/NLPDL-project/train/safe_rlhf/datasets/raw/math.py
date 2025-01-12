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
"""Scalable Correction dataset for supervised instruction fine-tuning2023-12-15 1:25 first commit"""
from __future__ import annotations
import json
# from datasets import load_dataset

from safe_rlhf.datasets.base import RawSample
from safe_rlhf.datasets.base import RawDataset

__all__ = [
    'MATHSupervisedDataset',
    'MATHPreferenceDataset',
]

# in /safe_rlhf/datasets/supervised.py, you can see how the PAD_token was added
class MATHSupervisedDataset(RawDataset):
    NAME: str = 'math-supervised'

    def __init__(self, path) -> None:  # noqa: ANN001
        self.path = path
        with open(self.path, encoding='utf-8') as f:
            self.data = json.load(f)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['problem']
        #input = ' '.join((data['question'], data['answer']))
        answer = data['solution']
        return RawSample(input=input, answer=answer)

    def __len__(self) -> int:
        return len(self.data)

class MATHPreferenceDataset(RawDataset):
    NAME: str = 'math-preference'
    
    def __init__(self, path) -> None:
        self.path = path
        with open(self.path, encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        input = data['question']
        better = data['prefix'] + data['correction']
        worse = data['prefix'] + data['last']
        return RawSample(
            input=input,
            answer=better,
            other_answer=worse,
            better=True
        )
    
    def __len__(self) -> int:
        return len(self.data)

