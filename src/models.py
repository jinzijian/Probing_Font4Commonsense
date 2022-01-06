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
from transformers import BertTokenizer, BertModel, BertConfig
import requests
from filelock import FileLock
from huggingface_hub import HfFolder, Repository
from transformers.utils.versions import importlib_metadata
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

configuration = XLMRobertaConfig()

class mbert_base(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(mbert_base, self).__init__()
        self.config = BertConfig()
        self.num_labels = num_labels
        self.input_dim = input_dim
        #self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.mbert = BertModel.from_pretrained("bert-base-multilingual-uncased")
        self.classifier = nn.Linear(self.input_dim, self.num_labels)
        self.dropout = nn.Dropout(p=0.3)
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
        args = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mbert(
            input_ids = input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )
        sequence_output = outputs[1]
        logits = self.classifier(sequence_output)
        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return loss, logits


class SeqClassification(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(SeqClassification, self).__init__()
        self.config = XLMRobertaConfig()
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.mbert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.classifier = nn.Linear(self.input_dim, self.num_labels)
        self.dropout = nn.Dropout(p=0.3)
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
        args = None,
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



class MixModel(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(MixModel, self).__init__()
        self.config = XLMRobertaConfig()
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base" )
        self.classifier = nn.Linear(self.input_dim * 2, self.num_labels)
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv1 = nn.Conv2d(3, 64, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 256, 5)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 768, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(768, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.mlp = nn.Linear(128, 768)
        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        input_imgs=None,
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
        seq_len = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #strings = self.tokenizer.batch_decode(input_ids)
        # input_img cnn -》 Batch * seqlength * 512
        x = F.relu(self.bn1(self.conv1(input_imgs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(self.conv3(x))
        x = torch.squeeze(x)
        # Batch * 512 --> LSTM  ---> Batch * seqlength * 768
        size = input_ids.size()
        x = torch.reshape(x, (size[0], 400, -1))  # cnn output
        #x = self.dropout(x)
        # get LSTM end  ---> Batch * 768
        rnn_output, (hn, cn) = self.lstm(x)
        hn = hn[2:]
        hn = self.mlp(hn)
        hn = self.dropout(hn)
        hn = torch.mean(hn, 0)
        outputs = self.roberta(
            input_ids = input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, :1, :]
        sequence_output = sequence_output.squeeze()
        sequence_output = torch.cat((sequence_output, hn), 1)
        logits = self.classifier(sequence_output)
        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, logits


    def get_parameter(self, base_lr) -> "Parameter":
        return [
            {"params": self.roberta.parameters(), "lr": 1.0*base_lr},
            {"params":self.conv1.parameters(),"lr":10*base_lr},
            {"params": self.conv2.parameters(), "lr": 10 * base_lr},
            {"params": self.conv3.parameters(), "lr": 10 * base_lr},
            {"params": self.lstm.parameters(), "lr": 10 * base_lr},
            {"params": self.classifier.parameters(), "lr": 10 * base_lr}
        ]


class mbert_cnn(nn.Module):
    def __init__(self, input_dim, num_labels):
        super(mbert_cnn, self).__init__()
        self.config = BertConfig()
        self.num_labels = num_labels
        self.input_dim = input_dim
        self.mbert = BertModel.from_pretrained("bert-base-multilingual-uncased")
        self.classifier = nn.Linear(self.input_dim * 2, self.num_labels)
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.linear1 = nn.Linear(800, 128)
        self.linear2 = nn.Linear(128, 128)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True)
        self.mlp = nn.Linear(128, 768)
        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        input_imgs=None,
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
        seq_len = None,
        maxlen = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #strings = self.tokenizer.batch_decode(input_ids)
        # input_img cnn -》 Batch * seqlength * 512
        x = F.relu(self.conv1(input_imgs))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = torch.squeeze(x)
        # Batch * 512 --> LSTM  ---> Batch * seqlength * 768
        size = input_ids.size()
        x = torch.reshape(x, (size[0], maxlen, 32*5*5))  # cnn output
        #x = self.dropout(x)
        # get LSTM end  ---> Batch * 768
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        rnn_output, (hn, cn) = self.lstm(x)
        hn = hn[2:]
        hn = self.mlp(hn)
        hn = self.dropout(hn)
        hn = torch.mean(hn, 0)
        outputs = self.mbert(
            input_ids = input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )
        sequence_output = outputs[1]
        sequence_output = torch.cat((sequence_output, hn), 1)
        logits = self.classifier(sequence_output)
        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, logits


    def get_parameter(self, base_lr) -> "Parameter":
        return [
            {"params": self.mbert.parameters(), "lr": 1.0*base_lr},
            {"params":self.conv1.parameters(),"lr":10*base_lr},
            {"params": self.conv2.parameters(), "lr": 10 * base_lr},
            {"params": self.conv3.parameters(), "lr": 10 * base_lr},
            {"params": self.linear1.parameters(), "lr": 10 * base_lr},
            {"params": self.mlp.parameters(), "lr": 10 * base_lr},
            {"params": self.linear2.parameters(), "lr": 10 * base_lr},
            {"params": self.lstm.parameters(), "lr": 10 * base_lr},
            {"params": self.classifier.parameters(), "lr": 10 * base_lr}
        ]