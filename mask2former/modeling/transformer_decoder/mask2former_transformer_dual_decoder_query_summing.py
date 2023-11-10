# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY

from .mask2former_transformer_dual_decoder import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP, _get_activation_fn

@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedDualTransformerDecoderQuerySumming(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes_anatomy: int,
        num_classes_pathology: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        
        ####### ANATOMICAL PREDICTION HEAD STARTS ###########
        ##### Alex start tomorrow here
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes_anatomy + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)


        ####### Pathology Strucure Setup starts here
        self.pe_layer_patho = PositionEmbeddingSine(N_steps, normalize=True)
        self.transformer_self_attention_layers_pathology = nn.ModuleList()
        self.transformer_cross_attention_layers_pathology = nn.ModuleList()
        self.transformer_ffn_layers_pathology = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers_pathology.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm
                )
            )

            self.transformer_cross_attention_layers_pathology.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers_pathology.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm
                )
            )

        self.decoder_norm_path = nn.LayerNorm(hidden_dim)
        #TODO: Make adjustable, might want to use different numbers of queries
        self.num_queries_path = num_queries

        #Learnable queries and Learnable PE Query emebeddings
        self.query_feat_path = nn.Embedding(self.num_queries_path, hidden_dim)
        self.query_embed_path = nn.Embedding(self.num_queries_path, hidden_dim)

        #Learnable Embedding for the levels. Always use 3 scales
        self.level_embed_path = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj_path = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj_path.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj_path[-1])
            else:
                #put Default value
                self.input_proj_path.append(nn.Sequential())

        #output FFNS
        if self.mask_classification:
            self.class_embed_path = nn.Linear(hidden_dim, num_classes_pathology + 1)
        self.mask_embed_path = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        ##Parameters for simple merging
        self.query_merging_params_ana = nn.ModuleList()
        #self.query_merging_params_patho = nn.ModuleList()
        for _ in range(self.num_feature_levels):
           # self.query_merging_params_ana.append(
           #     torch.nn.Embedding(num_queries, hidden_dim)
           #)
            #self.query_merging_params_patho.append(
            #    torch.nn.Embedding(self.num_queries_path, hidden_dim)
            #)
            self.query_merging_params_ana.append(
                torch.nn.Embedding(num_queries,1)
            )


    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        #TODO: ACTUALLY READ FROM CONFIG!!!!!!!!

        ret["num_classes_anatomy"] = 145
        ret["num_classes_pathology"] = 2
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        
        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        #Pathology Embeddings
        src_path = []
        pos_path = []
        size_list_path = []
        #Get positionel Embeddings, level embeddings and input projections 
        for i in range(self.num_feature_levels):
            size_list_path.append(x[i].shape[-2:])
            pos_path.append(self.pe_layer_patho(x[i], None).flatten(2))
            src_path.append(self.input_proj_path[i](x[i]).flatten(2) + self.level_embed_path.weight[i][None,:,None])

            # flatten NxCxHxW to HWxNxC
            pos_path[-1] = pos_path[-1].permute(2, 0, 1)
            src_path[-1] = src_path[-1].permute(2, 0, 1)

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        query_embed_path = self.query_embed_path.weight.unsqueeze(1).repeat(1, bs, 1)
        output_path = self.query_feat_path.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        predictions_class_path = []
        predictions_mask_path = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        outputs_class_path, outputs_mask_path, attn_mask_path = self.forward_prediction_heads(output_path, mask_features, attn_mask_target_size=size_list_path[0], dtype="Pathology")
        predictions_class_path.append(outputs_class_path)
        predictions_mask_path.append(outputs_mask_path)

        
        
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            #Perform for Pathologie. 
            attn_mask_path[torch.where(attn_mask_path.sum(-1) == attn_mask_path.shape[-1])] = False
            output_path = self.transformer_cross_attention_layers_pathology[i](
                output_path, src_path[level_index],
                memory_mask=attn_mask_path,
                memory_key_padding_mask=None,
                pos=pos_path[level_index], query_pos=query_embed_path
            )

            output_path = self.transformer_self_attention_layers_pathology[i](
                output_path, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed_path
            )

            output_path = self.transformer_ffn_layers_pathology[i](
                output_path
            )

            ###THIS IS ALL WE DO:
            #OUTPUT Anatomy: QxBSxHD (100X2X256)
            #OUTPUT Pathology: QXBSXHD (100X2X256)
            
            #output, output_path =  output + output_path * self.query_merging_params_ana[level_index].weight.unsqueeze(1).repeat(1, bs, 1), output_path + output * self.query_merging_params_patho[level_index].weight.unsqueeze(1).repeat(1, bs, 1)
            
            #Normalize and reshape to BSXQXH
            
            norm_output = output.detach() / output.detach().norm(dim=2)[:,:,None]
            norm_output = norm_output.transpose(0,1)
            norm_path = output_path.detach() / output_path.detach().norm(dim=2)[:,:, None]
            norm_path = norm_path.transpose(0,1)
            
            cosine_similarity_matrix = torch.bmm(norm_output, norm_path.transpose(1,2))
            weighted_anatomy_queries = torch.bmm(cosine_similarity_matrix,norm_output)

            #Reshape back to QXBSXHD
            weighted_anatomy_queries = weighted_anatomy_queries.transpose(0,1)

            output_path  = output_path + self.query_merging_params_ana[level_index].weight.unsqueeze(1).repeat(1, bs, 1) * weighted_anatomy_queries

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            outputs_class_path, outputs_mask_path, attn_mask_path = self.forward_prediction_heads(output_path, mask_features,  attn_mask_target_size=size_list_path[(i + 1) % self.num_feature_levels], dtype="Pathology")
            predictions_class_path.append(outputs_class_path)
            predictions_mask_path.append(outputs_mask_path)

        assert len(predictions_class) == self.num_layers + 1
        assert len(predictions_class_path) == self.num_layers +1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        out_path = {
            'pred_logits': predictions_class_path[-1],
            'pred_masks': predictions_mask_path[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class_path if self.mask_classification else None, predictions_mask_path
            )
        }
        return out, out_path

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, dtype="anatomy"):
        
        decoder_output = self.decoder_norm(output) if dtype == "anatomy" else self.decoder_norm_path(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output) if dtype == "anatomy" else self.class_embed_path(decoder_output)
        mask_embed = self.mask_embed(decoder_output) if dtype == "anatomy" else self.mask_embed_path(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]