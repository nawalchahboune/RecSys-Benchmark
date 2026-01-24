# -*- coding: utf-8 -*-
r"""
SASRecF
################################################
Un séquentiel type Transformer avec fusion adaptative (gating) des attributs d'items.
"""

# normalemenrt c'est HLLMRec mais pour éviter les configurations conflictuelles on le nomme SASRecF ici.

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer
from recbole.model.loss import BPRLoss
from recbole.utils import FeatureType


class SASRecF(SequentialRecommender):
    """Transformer séquentiel avec fusion adaptative des attributs (gate).
    Concatène et compresse les attributs puis applique un gate pour moduler la représentation item.
    """

    def __init__(self, config, dataset):
        super(SASRecF, self).__init__(config, dataset)

        # paramètres
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.selected_features = config.get("selected_features", [])
        self.pooling_mode = config.get("pooling_mode", "mean")
        self.device = config["device"]
        self.num_feature_field = sum(
            (
                1
                if dataset.field2type[field] != FeatureType.FLOAT_SEQ
                else dataset.num(field)
            )
            for field in self.selected_features
        )
        self.initializer_range = config.get("initializer_range", 0.02)
        self.loss_type = config.get("loss_type", "CE")

        # couches
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset,
            self.hidden_size,
            self.selected_features,
            self.pooling_mode,
            self.device,
        )

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

        # compression des features et gate
        in_dim = self.hidden_size * max(1, self.num_feature_field)
        self.feature_proj = nn.Linear(in_dim, self.hidden_size)
        self.gate_layer = nn.Linear(self.hidden_size, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

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

    def forward(self, item_seq, item_seq_len):
        # embeddings de base
        item_emb = self.item_embedding(item_seq)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # embeddings d'attributs
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding.get("item", None) if isinstance(sparse_embedding, dict) else sparse_embedding
        dense_embedding = dense_embedding.get("item", None) if isinstance(dense_embedding, dict) else dense_embedding

        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)

        if len(feature_table) > 0:
            feature_table = torch.cat(feature_table, dim=-2)  # concat sur le nombre de champs
            B, L, feat_num, H = feature_table.shape[0], feature_table.shape[1], feature_table.shape[-2], feature_table.shape[-1]
            feature_emb = feature_table.view(B, L, feat_num * H)
        else:
            # pas de features: zéro tensor [B, L, H] pour garder la forme
            feature_emb = torch.zeros_like(item_emb)

        # projection et gate
        feat_proj = self.feature_proj(feature_emb)
        gate = torch.sigmoid(self.gate_layer(feat_proj))
        fused = item_emb + gate * feat_proj

        # encoder
        input_emb = fused + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores