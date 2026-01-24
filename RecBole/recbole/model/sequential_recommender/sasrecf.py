# -*- coding: utf-8 -*-
"""
SASRecF (HLLMRec-MM)
Multi-gate, multi-source sequential ranker with optional logging
"""

import torch
from torch import nn
import logging

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer
from recbole.model.loss import BPRLoss
from recbole.utils import FeatureType


def get_logger(name="recbole", level=logging.INFO):
    """Local logger setup."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class SASRecF(SequentialRecommender):
    """
    Extensions:
    - ID embedding
    - Feature embedding (metadata)
    - Retrieval score embedding (optional)
    - Independent gates for each source
    - Debug logs (tensors + CUDA mem)
    """

    def __init__(self, config, dataset):
        super(SASRecF, self).__init__(config, dataset)
        print("="*60)
        print(f"MODEL SASRecF lancé avec config s7i7a:")
        print(f"  n_layers={config.get('n_layers')}, n_heads={config.get('n_heads')}, hidden_size={config.get('hidden_size')}, inner_size={config.get('inner_size')}")
        print(f"  selected_features={config.get('selected_features')}, pooling_mode={config.get('pooling_mode')}, loss_type={config.get('loss_type')}")
        print("="*60)


        # load parameters
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config.get("attn_dropout_prob", self.hidden_dropout_prob)
        self.hidden_act = config.get("hidden_act", "gelu")
        self.layer_norm_eps = config.get("layer_norm_eps", 1e-12)

        self.selected_features = config.get("selected_features", [])
        self.pooling_mode = config.get("pooling_mode", "mean")
        self.device = config["device"]
        self.num_feature_field = sum(
            (1 if dataset.field2type[field] != FeatureType.FLOAT_SEQ else dataset.num(field))
            for field in self.selected_features
        )

        self.initializer_range = config.get("initializer_range", 0.02)
        self.loss_type = config.get("loss_type", "CE")

        # logging flags
        self.debug_model = bool(config.get("debug_model", False))
        state = str(config.get("state", "INFO")).upper()
        level = logging.DEBUG if state == "DEBUG" else logging.INFO
        self.logger = get_logger(level=level)
        self._step = 0

        # embeddings
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset,
            self.hidden_size,
            self.selected_features,
            self.pooling_mode,
            self.device,
        )

        # feature compression to H
        in_dim = max(1, self.num_feature_field) * self.hidden_size
        self.feature_proj = nn.Linear(in_dim, self.hidden_size)

        # retrieval score projection (1 -> H)
        self.retr_proj = nn.Linear(1, self.hidden_size)

        # gates
        self.gate_id = nn.Linear(self.hidden_size, self.hidden_size)
        self.gate_feat = nn.Linear(self.hidden_size, self.hidden_size)
        self.gate_retr = nn.Linear(self.hidden_size, self.hidden_size)

        # transformer encoder (same API/style as original)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # loss
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # init
        self.apply(self._init_weights)
        self.other_parameter_name = ["feature_embed_layer"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _log_tensor(self, name, t):
        if not self.debug_model or t is None:
            return
        try:
            stats = {
                "shape": tuple(t.shape),
                "mean": float(t.mean().detach().cpu().item()),
                "std": float(t.std().detach().cpu().item()),
                "min": float(t.min().detach().cpu().item()),
                "max": float(t.max().detach().cpu().item()),
            }
        except Exception:
            stats = {"shape": tuple(t.shape)}
        self.logger.info(f"[{self.__class__.__name__}] {name}: {stats}")

    def _log_mem(self, device):
        if not self.debug_model:
            return
        if torch.cuda.is_available() and device.type == "cuda":
            a = torch.cuda.memory_allocated(device)
            r = torch.cuda.memory_reserved(device)
            self.logger.info(f"[{self.__class__.__name__}] CUDA mem: allocated={a/1e9:.2f}GB reserved={r/1e9:.2f}GB")

    def forward(self, item_seq, item_seq_len, retr_score=None):
        self._step += 1

        # ID embedding
        item_emb = self.item_embedding(item_seq)

        # position embedding (same style as original)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        pos_emb = self.position_embedding(position_ids)

        # feature embeddings
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_item = None
        dense_item = None
        if sparse_embedding is not None:
            sparse_item = sparse_embedding["item"] if isinstance(sparse_embedding, dict) else sparse_embedding
        if dense_embedding is not None:
            dense_item = dense_embedding["item"] if isinstance(dense_embedding, dict) else dense_embedding

        feats = None
        feats_list = []
        if sparse_item is not None:
            feats_list.append(sparse_item)
        if dense_item is not None:
            feats_list.append(dense_item)

        if len(feats_list) > 0:
            feats = torch.cat(feats_list, dim=-2)  # concat on field axis
            B, T, F, H = feats.shape
            feat_flat = feats.view(B, T, F * H)
            feat_emb = self.feature_proj(feat_flat)
        else:
            feat_emb = torch.zeros_like(item_emb)

        # retrieval score → embedding
        if retr_score is not None:
            retr_emb = self.retr_proj(retr_score.unsqueeze(-1))
        else:
            retr_emb = torch.zeros_like(item_emb)

        # gates
        gid = torch.sigmoid(self.gate_id(item_emb))
        gfeat = torch.sigmoid(self.gate_feat(feat_emb))
        gretr = torch.sigmoid(self.gate_retr(retr_emb))

        fused = gid * item_emb + gfeat * feat_emb + gretr * retr_emb

        # logs (début et toutes les 100 itérations)
        if self.debug_model & (self._step <= 3 or self._step % 100 == 0):
            self._log_tensor("item_emb", item_emb)
            self._log_tensor("feat_emb", feat_emb)
            self._log_tensor("retr_emb", retr_emb)
            self._log_tensor("gid", gid)
            self._log_tensor("gfeat", gfeat)
            self._log_tensor("gretr", gretr)
            self._log_tensor("fused", fused)
            self._log_mem(item_seq.device)

        # transformer pass
        x = fused + pos_emb
        x = self.LayerNorm(x)
        x = self.dropout(x)

        mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(x, mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)

        if self.debug_model and (self._step <= 3 or self._step % 100 == 0):
            self._log_tensor("seq_output", seq_output)

        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        try:
            retr = interaction["retr_score"]
        except Exception:
            retr = None
        seq_output = self.forward(item_seq, item_seq_len, retr)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            if self.debug_model and (self._step <= 3 or self._step % 100 == 0):
                self._log_tensor("pos_score", pos_score)
                self._log_tensor("neg_score", neg_score)
                self._log_tensor("loss", loss)
            return loss
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            if self.debug_model and (self._step <= 3 or self._step % 100 == 0):
                self._log_tensor("logits", logits)
                self._log_tensor("loss", loss)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        try:
            retr = interaction["retr_score"]
        except Exception:
            retr = None
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len, retr)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        try:
            retr = interaction["retr_score"]
        except Exception:
            retr = None
        seq_output = self.forward(item_seq, item_seq_len, retr)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores