from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cn_clip.clip import _tokenizer
from cn_clip.clip.configuration_bert import BertConfig
from cn_clip.clip.modeling_bert import BertModel
from typing import Optional
from .esresnet import ESResNeXtFBSP

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # text
                 vocab_size: int,
                 text_attention_probs_dropout_prob: float, 
                 text_hidden_act: str, 
                 text_hidden_dropout_prob: float, 
                 text_hidden_size: int,
                 text_initializer_range: float, 
                 text_intermediate_size: int, 
                 text_max_position_embeddings: int, 
                 text_num_attention_heads: int, 
                 text_num_hidden_layers: int, 
                 text_type_vocab_size: int,
                 tokenizer = _tokenizer,
                 # audio
                 n_fft: int = 2048,
                 hop_length: Optional[int] = 561,
                 win_length: Optional[int] = 1654,
                 window: Optional[str] = 'blackmanharris',
                 normalized: bool = True,
                 onesided: bool = True,
                 spec_height: int = -1,
                 spec_width: int = -1,
                 apply_attention: bool = True,
                #  multilabel: bool = True,
                 pretrained: Union[bool, str] = True,
                 ):
        super().__init__()

        self.audio = ESResNeXtFBSP(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=embed_dim,
            apply_attention=apply_attention,
            pretrained=False
        )

        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=text_hidden_size,
            num_hidden_layers=text_num_hidden_layers,
            num_attention_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            hidden_act=text_hidden_act,
            hidden_dropout_prob=text_hidden_dropout_prob,
            attention_probs_dropout_prob=text_attention_probs_dropout_prob,
            max_position_embeddings=text_max_position_embeddings,
            type_vocab_size=text_type_vocab_size,
            initializer_range=text_initializer_range,
            layer_norm_eps=1e-12,
        )
        self.bert = BertModel(self.bert_config)

        self.text_projection = nn.Parameter(torch.empty(text_hidden_size, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.tokenizer = tokenizer

        self.initialize_parameters()

    def initialize_parameters(self):
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.bert_config.hidden_size ** -0.5)

    @property
    def dtype(self):
        return self.text_projection.dtype

    def encode_audio(self, audio):
        # pass
        return self.audio(audio)

    def encode_text(self, text):
        pad_index = self.tokenizer.vocab['[PAD]']
        attn_mask = text.ne(pad_index).type(self.dtype)
        x = self.bert(text, attention_mask=attn_mask)[0].type(self.dtype) # [batch_size, seq_length, hidden_size]
        return x[:, 0, :] @ self.text_projection

    def forward(self, audio, text):
        assert audio is not None or text is not None, "text and audio cannot both be None!"

        if audio is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_audio(audio)
        audio_features = self.encode_audio(audio)
        text_features = self.encode_text(text)

        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return audio_features, text_features, self.logit_scale.exp()

    def get_similarity(self, audio, text):
        audio_features = self.encode_audio(audio)
        text_features = self.encode_text(text)

        # normalized features
        audio_features = audio_features / audio_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * audio_features @ text_features.t()
        logits_per_text = logits_per_audio.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_audio, logits_per_text


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        if isinstance(l, BertModel):
            l.to(torch.half)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def restore_model(model, clip_state_dict: dict, bert_state_dict: dict):
    merged_state_dict = {}

    # use clip_state_dict to initialize the audio encoder & logit scale
    if clip_state_dict is not None:
        for k, v in clip_state_dict.items():
            if k.startswith("visual") or k == "logit_scale":
                merged_state_dict[k] = v

    # use bert_state_dict to initialize the text encoder
    if bert_state_dict is not None:
        for k, v in bert_state_dict.items():
            if k.startswith("bert"):
                merged_state_dict[k] = v

    convert_weights(model)
    model.load_state_dict(merged_state_dict, strict=False)
    return model.eval()
