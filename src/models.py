import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import RobertaPreTrainedModel, XLMRobertaModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import copy
import fnmatch
import functools
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
from collections import OrderedDict, UserDict
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from enum import Enum
from functools import partial, wraps
from hashlib import sha256
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO, ContextManager, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
from zipfile import ZipFile, is_zipfile

import numpy as np
from packaging import version
from tqdm.auto import tqdm

import requests
from filelock import FileLock
from huggingface_hub import HfFolder, Repository
from transformers.utils.versions import importlib_metadata
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig

configuration = XLMRobertaConfig()
class SeqClassification(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(SeqClassification, self).__init__()
        self.config = XLMRobertaConfig()
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.classifier = nn.Linear(self.input_dim, self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids = input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, :1, :]
        sequence_output = sequence_output.squeeze()
        logits = self.classifier(sequence_output)
        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, logits
