from __future__ import annotations

from typing import Sequence

from .constants import MODEL_ARCH, MODEL_TENSOR, MODEL_TENSORS, TENSOR_NAMES

class TensorNameMap:
    mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {

        MODEL_TENSOR.TOKEN_EMBD: (
            "gpt_neox.embed_in",
            "transformer.wte",
            "transformer.word_embeddings",
            "word_embeddings",
            "model.embed_tokens",
            "tok_embeddings",
            "embeddings.word_embeddings",
            "language_model.embedding.word_embeddings",
            "wte",
            "transformer.embd.wte",
            "model.tok_embeddings",
            "model.embedding",
            "backbone.embedding",
            "backbone.embeddings",
            "transformer.in_out_embed",
            "embedding.word_embeddings",
            "transformer.token_embeddings",
            "shared",
        ),

        MODEL_TENSOR.TOKEN_TYPES: (
            "embeddings.token_type_embeddings",
        ),

        MODEL_TENSOR.TOKEN_EMBD_NORM: (
            "word_embeddings_layernorm",
            "embeddings.LayerNorm",
            "emb_ln",
            "transformer.norm",
        ),

        MODEL_TENSOR.POS_EMBD: (
            "transformer.wpe",
            "embeddings.position_embeddings",
            "wpe",
        ),

        MODEL_TENSOR.OUTPUT: (
            "embed_out",
            "lm_head",
            "output",
            "word_embeddings_for_head",
            "lm_head.linear",
            "output_layer",
        ),

        MODEL_TENSOR.OUTPUT_NORM: (
            "gpt_neox.final_layer_norm",
            "transformer.ln_f",
            "model.norm",
            "norm",
            "transformer.norm_f",
            "ln_f",
            "language_model.encoder.final_layernorm",
            "model.final_layernorm",
            "lm_head.ln",
            "model.norm_f",
            "backbone.norm_f",
            "transformer.rms_norm",
            "encoder.final_layernorm",
            "transformer.norm",
            "model.norm",
        ),

        MODEL_TENSOR.ROPE_FREQS: (
            "rope.freqs",
            "rotary_pos_emb.inv_freq",
        ),
    }

    block_mappings_cfg: dict[MODEL_TENSOR, tuple[str, ...]] = {

        MODEL_TENSOR.ATTN_NORM: (
            "gpt_neox.layers.{bid}.input_layernorm",
            "transformer.h.{bid}.ln_1",
            "transformer.blocks.{bid}.norm_1",
            "transformer.h.{bid}.input_layernorm",
            "h.{bid}.input_layernorm",
            "transformer.h.{bid}.ln_mlp",
            "model.layers.{bid}.input_layernorm",
            "layers.{bid}.attention_norm",
            "language_model.encoder.layers.{bid}.input_layernorm",
            "model.layers.{bid}.ln1",
            "h.{bid}.ln_1",
            "transformer.h.{bid}.ln",
            "model.layers.layers.{bid}.norm",
            "model.layers.{bid}.attention_norm",
            "model.layers.{bid}.norm",
            "backbone.layers.{bid}.norm",
            "transformer.decoder_layer.{bid}.rms_norm",
            "transformer.blocks.{bid}.norm_attn_norm.norm_1",
            "encoder.layers.{bid}.input_layernorm",
            "transformer.layers.{bid}.attn_norm",
        ),

        MODEL_TENSOR.ATTN_NORM_2: (
            "transformer.h.{bid}.ln_attn",
            "encoder.layer.{bid}.layer_norm_1",
        ),

        MODEL_TENSOR.ATTN_QKV: (
            "gpt_neox.layers.{bid}.attention.query_key_value",
            "transformer.h.{bid}.attn.c_attn",
            "transformer.blocks.{bid}.attn.Wqkv",
            "transformer.blocks.{bid}.norm_attn_norm.attn.Wqkv",
            "transformer.h.{bid}.self_attention.query_key_value",
            "h.{bid}.self_attention.query_key_value",
            "language_model.encoder.layers.{bid}.self_attention.query_key_value",
            "model.layers.{bid}.self_attn.query_key_value",
            "h.{bid}.attn.c_attn",
            "transformer.h.{bid}.mixer.Wqkv",
            "encoder.layers.{bid}.attn.Wqkv",
            "model.layers.{bid}.self_attn.qkv_proj",
            "encoder.layers.{bid}.self_attention.query_key_value",
            "transformer.layers.{bid}.attn.qkv_proj",
        ),

        MODEL_TENSOR.ATTN_Q: (
            "model.layers.{bid}.self_attn.q_proj",
            "layers.{bid}.attention.wq",
            "encoder.layer.{bid}.attention.self.query",
            "transformer.h.{bid}.attn.q_proj",
            "model.layers.layers.{bid}.self_attn.q_proj",
            "model.layers.{bid}.attention.wq",
            "transformer.decoder_layer.{bid}.multi_head_attention.query",
            "transformer.h.{bid}.attn.attention.q_proj",
        ),

        MODEL_TENSOR.ATTN_K: (
            "model.layers.{bid}.self_attn.k_proj",
            "layers.{bid}.attention.wk",
            "encoder.layer.{bid}.attention.self.key",
            "transformer.h.{bid}.attn.k_proj",
            "transformer.h.{bid}.attn.k",
            "model.layers.layers.{bid}.self_attn.k_proj",
            "model.layers.{bid}.attention.wk",
            "transformer.decoder_layer.{bid}.multi_head_attention.key",
            "transformer.h.{bid}.attn.attention.k_proj",
        ),

        MODEL_TENSOR.ATTN_V: (
            "model.layers.{bid}.self_attn.v_proj",
            "layers.{bid}.attention.wv",
            "encoder.layer.{bid}.attention.self.value",
            "transformer.h.{bid}.attn.v_proj",
            "transformer.h.{bid}.attn.v",
            "model.layers.layers.{bid}.self_attn.v_proj",
            "model.layers.{bid}.attention.wv",
            "transformer.decoder_layer.{bid}.multi_head_attention.value",
            "transformer.h.{bid}.attn.attention.v_proj",
        ),

        MODEL_TENSOR.ATTN_OUT: (
            "gpt_neox.layers.{bid}.attention.dense",
            "transformer.h.{bid}.attn.c_proj",
            "transformer.blocks.{bid}.attn.out_proj",
            "transformer.h.{bid}.self_attention.dense",
            "h.{bid}.self_attention.dense",
            "model.layers.{bid}.self_attn.o_proj",
            "layers.{bid}.attention.wo",
            "encoder.layer.{bid}.attention.output.dense",
            "transformer.h.{bid}.attn.out_proj",
            "language_model.encoder.layers.{bid}.self_attention.dense",
            "model.layers.{bid}.self_attn.dense",
            "h.{bid}.attn.c_proj",
            "transformer.h.{bid}.mixer.out_proj",
            "model.layers.layers.{bid}.self_attn.o_proj",
            "model.layers.{bid}.attention.wo",
            "encoder.layers.{bid}.attn.out_proj",
            "transformer.decoder_layer.{bid}.multi_head_attention.linear",
            "transformer.blocks.{bid}.norm_attn_norm.attn.out_proj",
            "encoder.layers.{bid}.self_attention.dense",
            "transformer.layers.{bid}.attn.out_proj",
            "transformer.h.{bid}.attn.attention.out_proj",
        ),

        MODEL_TENSOR.ATTN_OUT_NORM: (
            "encoder.layer.{bid}.attention.output.LayerNorm",
            "encoder.layers.{bid}.norm1",
            "transformer.decoder_layer.{bid}.rms_norm_1",
            "transformer.blocks.{bid}.norm_attn_norm.norm_2",
        ),

        MODEL_TENSOR.ATTN_POST_NORM: (
            "model.layers.{bid}.post_attention_layernorm",
        ),

        MODEL_TENSOR.ATTN_ROT_EMBD: (
            "model.layers.{bid}.self_attn.rotary_emb.inv_freq",
            "layers.{bid}.attention.inner_attention.rope.freqs",
            "model.layers.layers.{bid}.self_attn.rotary_emb.inv_freq",
            "transformer.h.{bid}.attn.rotary_emb.inv_freq",
        ),

        MODEL_TENSOR.FFN_NORM: (
            "gpt_neox.layers.{bid}.post_attention_layernorm",
            "transformer.h.{bid}.ln_2",
            "h.{bid}.post_attention_layernorm",
            "transformer.blocks.{bid}.norm_2",
            "model.layers.{bid}.post_attention_layernorm",
            "layers.{bid}.ffn_norm",
            "language_model.encoder.layers.{bid}.post_attention_layernorm",
            "model.layers.{bid}.ln2",
            "h.{bid}.ln_2",
            "model.layers.{bid}.ffn_norm",
            "transformer.decoder_layer.{bid}.rms_norm_2",
            "encoder.layers.{bid}.post_attention_layernorm",
            "transformer.layers.{bid}.ffn_norm",
        ),

        MODEL_TENSOR.FFN_PRE_NORM: (
            "model.layers.{bid}.pre_feedforward_layernorm",
        ),

        MODEL_TENSOR.FFN_POST_NORM: (
            "model.layers.{bid}.post_feedforward_layernorm",
        ),

        MODEL_TENSOR.FFN_GATE_INP: (
            "layers.{bid}.feed_forward.gate",
            "model.layers.{bid}.block_sparse_moe.gate",
            "model.layers.{bid}.mlp.gate",
            "transformer.decoder_layer.{bid}.router",
            "transformer.blocks.{bid}.ffn.router.layer",
        ),

        MODEL_TENSOR.FFN_GATE_INP_SHEXP: (
            "model.layers.{bid}.mlp.shared_expert_gate",
        ),

        MODEL_TENSOR.FFN_UP: (
            "gpt_neox.layers.{bid}.mlp.dense_h_to_4h",
            "transformer.h.{bid}.mlp.c_fc",
            "transformer.blocks.{bid}.ffn.up_proj",
            "transformer.h.{bid}.mlp.dense_h_to_4h",
            "h.{bid}.mlp.dense_h_to_4h",
            "model.layers.{bid}.mlp.up_proj",
            "layers.{bid}.feed_forward.w3",
            "encoder.layer.{bid}.intermediate.dense",
            "transformer.h.{bid}.mlp.fc_in",
            "transformer.h.{bid}.mlp.linear_3",
            "language_model.encoder.layers.{bid}.mlp.dense_h_to_4h",
            "model.layers.{bid}.mlp.dense_h_to_4h",
            "transformer.h.{bid}.mlp.w1",
            "h.{bid}.mlp.c_fc",
            "transformer.h.{bid}.mlp.fc1",
            "model.layers.{bid}.mlp.fc1",
            "model.layers.{bid}.mlp.gate_up_proj",
            "model.layers.layers.{bid}.mlp.up_proj",
            "model.layers.{bid}.feed_forward.w3",
            "encoder.layers.{bid}.mlp.fc11",
            "model.layers.{bid}.mlp.c_fc",
            "encoder.layer.{bid}.mlp.gated_layers_v",
            "model.layers.{bid}.residual_mlp.w3",
            "encoder.layers.{bid}.mlp.dense_h_to_4h",
            "transformer.h.{bid}.mlp.c_fc_1",
        ),

        MODEL_TENSOR.FFN_UP_EXP: (
            "layers.{bid}.feed_forward.experts.w3",
            "transformer.decoder_layer.{bid}.moe.linear_v",
            "transformer.blocks.{bid}.ffn.experts.mlp.v1",
            "model.layers.{bid}.mlp.experts.up_proj",
        ),

        MODEL_TENSOR.FFN_UP_SHEXP: (
            "model.layers.{bid}.mlp.shared_expert.up_proj",
            "model.layers.{bid}.mlp.shared_experts.up_proj",
        ),

        MODEL_TENSOR.FFN_ACT: (
            "transformer.blocks.{bid}.ffn.act",
        ),

        MODEL_TENSOR.FFN_GATE: (
            "model.layers.{bid}.mlp.gate_proj",
            "layers.{bid}.feed_forward.w1",
            "transformer.h.{bid}.mlp.w2",
            "transformer.h.{bid}.mlp.c_fc2",
            "model.layers.layers.{bid}.mlp.gate_proj",
            "model.layers.{bid}.feed_forward.w1",
            "encoder.layers.{bid}.mlp.fc12",
            "encoder.layer.{bid}.mlp.gated_layers_w",
            "transformer.h.{bid}.mlp.linear_1",
            "model.layers.{bid}.residual_mlp.w1",
            "transformer.h.{bid}.mlp.c_fc_0",
        ),

        MODEL_TENSOR.FFN_GATE_EXP: (
            "layers.{bid}.feed_forward.experts.w1",
            "transformer.decoder_layer.{bid}.moe.linear",
            "transformer.blocks.{bid}.ffn.experts.mlp.w1",
            "model.layers.{bid}.mlp.experts.gate_proj",
        ),

        MODEL_TENSOR.FFN_GATE_SHEXP: (
            "model.layers.{bid}.mlp.shared_expert.gate_proj",
            "model.layers.{bid}.mlp.shared_experts.gate_proj",
        ),

        MODEL_TENSOR.FFN_DOWN: (
            "gpt_neox.layers.{bid}.mlp.dense_4h_to_h",
            "transformer.h.{bid}.mlp.c_proj",
            "transformer.blocks.{bid}.ffn.down_proj",
            "transformer.h.{bid}.mlp.dense_4h_to_h",
            "h.{bid}.mlp.dense_4h_to_h",
            "model.layers.{bid}.mlp.down_proj",
            "layers.{bid}.feed_forward.w2",
            "encoder.layer.{bid}.output.dense",
            "transformer.h.{bid}.mlp.fc_out",
            "language_model.encoder.layers.{bid}.mlp.dense_4h_to_h",
            "model.layers.{bid}.mlp.dense_4h_to_h",
            "h.{bid}.mlp.c_proj",
            "transformer.h.{bid}.mlp.fc2",
            "model.layers.{bid}.mlp.fc2",
            "model.layers.layers.{bid}.mlp.down_proj",
            "model.layers.{bid}.feed_forward.w2",
            "encoder.layers.{bid}.mlp.fc2",
            "model.layers.{bid}.mlp.c_proj",
            "encoder.layer.{bid}.mlp.wo",
            "transformer.layers.{bid}.ffn.proj_2",
            "model.layers.{bid}.residual_mlp.w2",
            "encoder.layer.{bid}.mlp.down_layer",
            "encoder.layers.{bid}.mlp.dense_4h_to_h",
            "model.layers.h.{bid}.mlp.c_proj",
        ),

        MODEL_TENSOR.FFN_DOWN_EXP: (
            "layers.{bid}.feed_forward.experts.w2",
            "transformer.decoder_layer.{bid}.moe.linear_1",
            "transformer.blocks.{bid}.ffn.experts.mlp.w2",
            "model.layers.{bid}.mlp.experts.down_proj",
        ),

        MODEL_TENSOR.FFN_DOWN_SHEXP: (
            "model.layers.{bid}.mlp.shared_expert.down_proj",
            "model.layers.{bid}.mlp.shared_experts.down_proj",
        ),

        MODEL_TENSOR.ATTN_Q_NORM: (
            "language_model.encoder.layers.{bid}.self_attention.q_layernorm",
            "model.layers.{bid}.self_attn.q_layernorm",
            "model.layers.{bid}.self_attn.q_norm",
            "transformer.blocks.{bid}.attn.q_ln",
            "encoder.layer.{bid}.attention.self.layer_norm_q",
            "transformer.layers.{bid}.attn.q_norm",
        ),

        MODEL_TENSOR.ATTN_K_NORM: (
            "language_model.encoder.layers.{bid}.self_attention.k_layernorm",
            "model.layers.{bid}.self_attn.k_layernorm",
            "model.layers.{bid}.self_attn.k_norm",
            "transformer.blocks.{bid}.attn.k_ln",
            "encoder.layer.{bid}.attention.self.layer_norm_k",
            "transformer.layers.{bid}.attn.k_norm",
        ),

        MODEL_TENSOR.ROPE_FREQS: (
            "language_model.encoder.layers.{bid}.self_attention.rotary_emb.inv_freq",
        ),

        MODEL_TENSOR.LAYER_OUT_NORM: (
            "encoder.layer.{bid}.output.LayerNorm",
            "encoder.layers.{bid}.norm2",
            "transformer.decoder_layer.{bid}.rms_norm_3",
            "encoder.layer.{bid}.mlp.layernorm",
            "encoder.layer.{bid}.layer_norm_2"
        ),

        MODEL_TENSOR.SSM_IN: (
            "model.layers.{bid}.in_proj",
            "backbone.layers.{bid}.mixer.in_proj",
        ),

        MODEL_TENSOR.SSM_CONV1D: (
            "model.layers.{bid}.conv1d",
            "backbone.layers.{bid}.mixer.conv1d",
        ),

        MODEL_TENSOR.SSM_X: (
            "model.layers.{bid}.x_proj",
            "backbone.layers.{bid}.mixer.x_proj",
        ),

        MODEL_TENSOR.SSM_DT: (
            "model.layers.{bid}.dt_proj",
            "backbone.layers.{bid}.mixer.dt_proj",
        ),

        MODEL_TENSOR.SSM_A: (
            "model.layers.{bid}.A_log",
            "backbone.layers.{bid}.mixer.A_log",
        ),

        MODEL_TENSOR.SSM_D: (
            "model.layers.{bid}.D",
            "backbone.layers.{bid}.mixer.D",
        ),

        MODEL_TENSOR.SSM_OUT: (
            "model.layers.{bid}.out_proj",
            "backbone.layers.{bid}.mixer.out_proj",
        ),

        MODEL_TENSOR.ATTN_Q_A: (
            "model.layers.{bid}.self_attn.q_a_proj",
        ),

        MODEL_TENSOR.ATTN_Q_B: (
            "model.layers.{bid}.self_attn.q_b_proj",
        ),

        MODEL_TENSOR.ATTN_KV_A_MQA: (
            "model.layers.{bid}.self_attn.kv_a_proj_with_mqa",
        ),

        MODEL_TENSOR.ATTN_KV_B: (
            "model.layers.{bid}.self_attn.kv_b_proj",
        ),

        MODEL_TENSOR.ATTN_Q_A_NORM: (
            "model.layers.{bid}.self_attn.q_a_layernorm",
        ),

        MODEL_TENSOR.ATTN_KV_A_NORM: (
            "model.layers.{bid}.self_attn.kv_a_layernorm",
        ),

        MODEL_TENSOR.ATTN_SUB_NORM: (
            "model.layers.{bid}.self_attn.inner_attn_ln",
        ),

        MODEL_TENSOR.FFN_SUB_NORM: (
            "model.layers.{bid}.mlp.ffn_layernorm",
        ),

        MODEL_TENSOR.DEC_ATTN_NORM: (
            "decoder.block.{bid}.layer.0.layer_norm",
        ),

        MODEL_TENSOR.DEC_ATTN_Q: (
            "decoder.block.{bid}.layer.0.SelfAttention.q",
        ),

        MODEL_TENSOR.DEC_ATTN_K: (
            "decoder.block.{bid}.layer.0.SelfAttention.k",
        ),

        MODEL_TENSOR.DEC_ATTN_V: (
            "decoder.block.{bid}.layer.0.SelfAttention.v",
        ),

        MODEL_TENSOR.DEC_ATTN_OUT: (
            "decoder.block.{bid}.layer.0.SelfAttention.o",
        ),

        MODEL_TENSOR.DEC_ATTN_REL_B: (
            "decoder.block.{bid}.layer.0.SelfAttention.relative_attention_bias",
        ),

        MODEL_TENSOR.DEC_CROSS_ATTN_NORM: (
            "decoder.block.{bid}.layer.1.layer_norm",
        ),

        MODEL_TENSOR.DEC_CROSS_ATTN_Q: (
            "decoder.block.{bid}.layer.1.EncDecAttention.q",
        ),

        MODEL_TENSOR.DEC_CROSS_ATTN_K: (
            "decoder.block.{bid}.layer.1.EncDecAttention.k",
        ),

        MODEL_TENSOR.DEC_CROSS_ATTN_V: (
            "decoder.block.{bid}.layer.1.EncDecAttention.v",
        ),

        MODEL_TENSOR.DEC_CROSS_ATTN_OUT: (
            "decoder.block.{bid}.layer.1.EncDecAttention.o",
        ),

        MODEL_TENSOR.DEC_CROSS_ATTN_REL_B: (
            "decoder.block.{bid}.layer.1.EncDecAttention.relative_attention_bias",
        ),

        MODEL_TENSOR.DEC_FFN_NORM: (
            "decoder.block.{bid}.layer.2.layer_norm",
        ),

        MODEL_TENSOR.DEC_FFN_GATE: (
            "decoder.block.{bid}.layer.2.DenseReluDense.wi_0",
        ),

        MODEL_TENSOR.DEC_FFN_UP: (
            "decoder.block.{bid}.layer.2.DenseReluDense.wi",
            "decoder.block.{bid}.layer.2.DenseReluDense.wi_1",
        ),

        MODEL_TENSOR.DEC_FFN_DOWN: (
            "decoder.block.{bid}.layer.2.DenseReluDense.wo",
        ),

        MODEL_TENSOR.DEC_OUTPUT_NORM: (
            "decoder.final_layer_norm",
        ),

        MODEL_TENSOR.ENC_ATTN_NORM: (
            "encoder.block.{bid}.layer.0.layer_norm",
        ),

        MODEL_TENSOR.ENC_ATTN_Q: (
            "encoder.block.{bid}.layer.0.SelfAttention.q",
        ),

        MODEL_TENSOR.ENC_ATTN_K: (
            "encoder.block.{bid}.layer.0.SelfAttention.k",
        ),

        MODEL_TENSOR.ENC_ATTN_V: (
            "encoder.block.{bid}.layer.0.SelfAttention.v",
        ),

        MODEL_TENSOR.ENC_ATTN_OUT: (
            "encoder.block.{bid}.layer.0.SelfAttention.o",
        ),

        MODEL_TENSOR.ENC_ATTN_REL_B: (
            "encoder.block.{bid}.layer.0.SelfAttention.relative_attention_bias",
        ),

        MODEL_TENSOR.ENC_FFN_NORM: (
            "encoder.block.{bid}.layer.1.layer_norm",
        ),

        MODEL_TENSOR.ENC_FFN_GATE: (
            "encoder.block.{bid}.layer.1.DenseReluDense.wi_0",
        ),

        MODEL_TENSOR.ENC_FFN_UP: (
            "encoder.block.{bid}.layer.1.DenseReluDense.wi",
            "encoder.block.{bid}.layer.1.DenseReluDense.wi_1",
        ),

        MODEL_TENSOR.ENC_FFN_DOWN: (
            "encoder.block.{bid}.layer.1.DenseReluDense.wo",
        ),

        MODEL_TENSOR.ENC_OUTPUT_NORM: (
            "encoder.final_layer_norm",
        ),
    }

    arch_block_mappings_cfg: dict[MODEL_ARCH, dict[MODEL_TENSOR, tuple[str, ...]]] = {
        MODEL_ARCH.ARCTIC: {
            MODEL_TENSOR.FFN_NORM: (
                "model.layers.{bid}.residual_layernorm",
            ),
            MODEL_TENSOR.FFN_NORM_EXP: (
                "model.layers.{bid}.post_attention_layernorm",
            ),
        },
    }

    mapping: dict[str, tuple[MODEL_TENSOR, str]]

    def __init__(self, arch: MODEL_ARCH, n_blocks: int):
        self.mapping = {}
        for tensor, keys in self.mappings_cfg.items():
            if tensor not in MODEL_TENSORS[arch]:
                continue
            tensor_name = TENSOR_NAMES[tensor]
            self.mapping[tensor_name] = (tensor, tensor_name)
            for key in keys:
                self.mapping[key] = (tensor, tensor_name)
        if arch in self.arch_block_mappings_cfg:
            self.block_mappings_cfg.update(self.arch_block_mappings_cfg[arch])
        for bid in range(n_blocks):
            for tensor, keys in self.block_mappings_cfg.items():
                if tensor not in MODEL_TENSORS[arch]:
                    continue

                tensor_name = TENSOR_NAMES[tensor].format(bid = bid)
                self.mapping[tensor_name] = (tensor, tensor_name)
                for key in keys:
                    key = key.format(bid = bid)
                    self.mapping[key] = (tensor, tensor_name)

    def get_type_and_name(self, key: str, try_suffixes: Sequence[str] = ()) -> tuple[MODEL_TENSOR, str] | None:
        result = self.mapping.get(key)
        if result is not None:
            return result
        for suffix in try_suffixes:
            if key.endswith(suffix):
                result = self.mapping.get(key[:-len(suffix)])
                if result is not None:
                    return result[0], result[1] + suffix
        return None

    def get_name(self, key: str, try_suffixes: Sequence[str] = ()) -> str | None:
        result = self.get_type_and_name(key, try_suffixes = try_suffixes)
        if result is None:
            return None
        return result[1]

    def get_type(self, key: str, try_suffixes: Sequence[str] = ()) -> MODEL_TENSOR | None:
        result = self.get_type_and_name(key, try_suffixes = try_suffixes)
        if result is None:
            return None
        return result[0]

    def __getitem__(self, key: str) -> str:
        try:
            return self.mapping[key][1]
        except KeyError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self.mapping

    def __repr__(self) -> str:
        return repr(self.mapping)

def get_tensor_name_map(arch: MODEL_ARCH, n_blocks: int) -> TensorNameMap:
    return TensorNameMap(arch, n_blocks)