"""
@author: Yanzuo Lu
@email: luyz5@mail2.sysu.edu.cn
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock

from .xf import FrozenCLIPImageEmbedder
import torch
torch.autograd.set_detect_anomaly(True)
import uuid

class CrossAttnFirstTransformerBlock(BasicTransformerBlock):
    sample_counter = 0

    def __init__(self, *args, block_idx=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_idx = block_idx  # 블록 인덱스 저장
        self.step_counter = 0  

    def get_attention_scores(self, query, key, attention_module, attention_mask=None):
        # Reshape query and key to [batch_size, num_heads, seq_length, head_dim]
        batch_size, seq_length, embed_dim = query.shape
        num_heads = attention_module.heads 
        head_dim = embed_dim // num_heads
        query = query.view(batch_size, seq_length, num_heads, head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, num_heads, head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-1, -2))  # [B, num_heads, query_length, key_length]
        
        # Apply softmax to get attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        return attn_probs

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        query_pos: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        # 1. Cross-Attention
        if self.attn2 is not None:
            hidden_states = hidden_states.clone() + query_pos
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            # Perform attention
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states.clone()

            # Get attention scores for attn2
            query = self.attn2.to_q(norm_hidden_states)
            key = self.attn2.to_k(encoder_hidden_states)

            # Debugging - check tensor shapes
            #print(f"attn2 query shape: {query.shape}")
            #print(f"attn2 key shape: {key.shape}")

            attn_weights_2 = self.get_attention_scores(query, key, self.attn2)

            # Save the attention map for attn2

            batch_size = query.shape[0]
            for idx_in_batch in range(batch_size):
                save_attention_map_total_query(
                    attn_weights_2,
                    "attn2",
                    step=self.step_counter,
                    block_idx=self.block_idx,
                    sample_idx=CrossAttnFirstTransformerBlock.sample_counter + idx_in_batch,
                    input_shape=(256, 256),
                    sample_batch_idx=idx_in_batch
                )

        # 2. Self-Attention
        hidden_states = hidden_states.clone() + query_pos
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # Debugging - check tensor shapes before attention
        #print(f"encoder_hidden_states shape: {encoder_hidden_states.shape if encoder_hidden_states is not None else 'None'}")
        #print(f"norm_hidden_states shape: {norm_hidden_states.shape}")

        # Perform attention
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states.clone()

        # Get attention scores for attn1
        query = self.attn1.to_q(norm_hidden_states)

        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
            encoder_hidden_states = nn.Linear(1024, 768).to(hidden_states.device)(encoder_hidden_states)

        key = self.attn1.to_k(encoder_hidden_states if encoder_hidden_states is not None else norm_hidden_states)
        # Debugging - check tensor shapes after query and key
        #print(f"attn1 query shape: {query.shape}")
        #print(f"attn1 key shape: {key.shape}")

        attn_weights_1 = self.get_attention_scores(query, key, self.attn1)

        batch_size = query.shape[0]
        for idx_in_batch in range(batch_size):
            save_attention_map_total_query(
                attn_weights_1,
                "attn1",
                step=self.step_counter,
                block_idx=self.block_idx,
                sample_idx=CrossAttnFirstTransformerBlock.sample_counter + idx_in_batch,
                input_shape=(256, 256),
                sample_batch_idx=idx_in_batch
            )

        # Step counter 증가
        self.step_counter += 1
        # sample_counter 증가
        CrossAttnFirstTransformerBlock.sample_counter += batch_size

        # Feed-forward and other operations follow...
        norm_hidden_states = self.norm3(hidden_states)
        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states.clone()

        return hidden_states

class Decoder(nn.Module):
    def __init__(self, n_ctx, ctx_dim, heads, depth, last_norm, img_size,
                 embed_dim, depths, pose_query, pose_channel):
        super().__init__()
        self.last_norm = last_norm
        self.pose_query = pose_query
        self.pose_channel = pose_channel
        self.ctx_dim = ctx_dim
        self.depth = depth

        if self.depth > 0:
            n_layers = len(depths)
            embed_dim = embed_dim * 2 ** (n_layers - 1)

            if not self.pose_query:
                self.query_feat = nn.Parameter(torch.zeros(n_ctx, ctx_dim))
                nn.init.normal_(self.query_feat, std=0.02)
            else:
                self.decoder_fc = nn.Linear(pose_channel, ctx_dim, bias=False)

            self.pos_embed = nn.Parameter(torch.zeros(n_ctx, ctx_dim))
            nn.init.normal_(self.pos_embed, std=0.02)

            self.blocks = []
            for idx in range(depth):  # idx를 사용하여 반복문 실행
                self.blocks.append(CrossAttnFirstTransformerBlock(
                    dim=ctx_dim,
                    num_attention_heads=heads,
                    attention_head_dim=ctx_dim // heads,
                    cross_attention_dim=embed_dim,
                    block_idx=idx  # idx를 블록 인덱스로 전달
                ))
            self.blocks = nn.ModuleList(self.blocks)

            if not self.last_norm:
                H, W = img_size[0] // 32, img_size[1] // 32
                self.kv_pos_embed = nn.Parameter(torch.zeros(1, H * W, embed_dim))
                nn.init.normal_(self.kv_pos_embed, std=0.02)

            # enable xformers
            def fn_recursive_set_mem_eff(module: torch.nn.Module):
                if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                    module.set_use_memory_efficient_attention_xformers(True, attention_op=None)

                for child in module.children():
                    fn_recursive_set_mem_eff(child)

            for module in self.children():
                if isinstance(module, torch.nn.Module):
                    fn_recursive_set_mem_eff(module)
        elif self.depth == 0:
            self.clip_model = FrozenCLIPImageEmbedder()
        elif self.depth == -2:
            n_layers = len(depths)
            embed_dim = embed_dim * 2 ** (n_layers - 1)
            self.decoder_fc = nn.Linear(embed_dim, ctx_dim, bias=False)

    def forward(self, x, features, pose_features):
        if self.depth > 0:
            if self.last_norm:
                B, C = x.shape
                encoder_hidden_states = x.unsqueeze(1)
            else:
                B, L, C = features[-1].shape
                encoder_hidden_states = features.pop()
                kv_pos_embed = self.kv_pos_embed.expand(B, -1, -1)
                encoder_hidden_states = encoder_hidden_states + kv_pos_embed

            if self.pose_query:
                hidden_states = pose_features.pop()
                if self.training:
                    hidden_states = hidden_states.reshape(B * 2, self.pose_channel, -1).permute(0, 2, 1)
                    pos_embed = self.pos_embed.expand(B * 2, -1, -1)
                    encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states])
                else:
                    hidden_states = hidden_states.reshape(B, self.pose_channel, -1).permute(0, 2, 1)
                    pos_embed = self.pos_embed.expand(B, -1, -1)

                hidden_states = self.decoder_fc(hidden_states)
            else:
                hidden_states = self.query_feat.expand(B, -1, -1)
                pos_embed = self.pos_embed.expand(B, -1, -1)

            for blk in self.blocks:
                hidden_states = blk(hidden_states, pos_embed, encoder_hidden_states=encoder_hidden_states)
            return hidden_states
        elif self.depth == 0:
            x = x * 0.5 + 0.5
            x = x - torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(dtype=x.dtype, device=x.device)
            x = x / torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(dtype=x.dtype, device=x.device)
            return self.clip_model(x)
        elif self.depth == -1:
            encoder_hidden_states = features.pop()
            encoder_hidden_states = encoder_hidden_states * 0.
            encoder_hidden_states = encoder_hidden_states.mean(dim=2, keepdim=True).expand(-1, -1, self.ctx_dim)
            return encoder_hidden_states
        elif self.depth == -2:
            encoder_hidden_states = features.pop()
            encoder_hidden_states = self.decoder_fc(encoder_hidden_states)
            return encoder_hidden_states


# Visualize function with resizing for visualization
import matplotlib.pyplot as plt
import numpy as np
import os
def match_size_with_input(attn_map, input_shape):
    """
    Resize attention map to match input image dimensions
    """
    from PIL import Image
    #print(f"attn_map_shape: {attn_map.shape}")  # Added this for debugging
    
    # Normalize and convert attention map to uint8 for visualization
    attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
    attn_map = attn_map.astype(np.uint8)

    attn_map_resized = Image.fromarray(attn_map)  # Convert to PIL image
    attn_map_resized = attn_map_resized.resize(input_shape, Image.BILINEAR)
    
    return np.array(attn_map_resized)
def save_attention_map_total_query(
    attn_weights,
    layer_name,
    step,
    block_idx=0,
    sample_idx=0,
    save_dir="attention_maps_total_query",
    input_shape=(256, 256),
    sample_batch_idx=0
):
    import os
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Debugging: print attn_weights shape
    #print(f"attn_weights shape: {attn_weights.shape}")
    
    # Get the attention weights for the specific sample and average over heads
    attn_weights_sample = attn_weights[sample_batch_idx].mean(dim=0).detach().cpu().numpy()    # Debugging: print attn_weights_sample shape
    #print(f"attn_weights_sample shape: {attn_weights_sample.shape}")
    
    # Ensure attn_weights_sample is 2D
    if attn_weights_sample.ndim == 2:
        query_length, key_length = attn_weights_sample.shape
    elif attn_weights_sample.ndim == 1:
        # In case it's 1D, reshape accordingly
        query_length = attn_weights_sample.shape[0]
        key_length = 1
        attn_weights_sample = attn_weights_sample[:, np.newaxis]
    else:
        raise ValueError(f"Unexpected attn_weights_sample shape: {attn_weights_sample.shape}")
    
    # Calculate H_query, W_query, H_key, W_key
    H_query = W_query = int(np.sqrt(query_length))
    H_key = W_key = int(np.sqrt(key_length))
    
    # Verify that the calculated H and W are correct
    assert H_query * W_query == query_length, f"Invalid H_query and W_query: {H_query}, {W_query}"
    assert H_key * W_key == key_length, f"Invalid H_key and W_key: {H_key}, {W_key}"
    
    # Reshape attention weights
    attn_weights_reshaped = attn_weights_sample.reshape(H_query, W_query, H_key, W_key)
    
    # Sum over query dimensions to get total attention on key positions
    attn_map = attn_weights_reshaped.sum(axis=(0, 1))  # Shape: [H_key, W_key]
    
    # Normalize the attention map
    attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map) + 1e-8)
    
    # Convert to float32 for cv2.resize
    attn_map = attn_map.astype(np.float32)
    
    # Resize attention map to input image size
    attn_map_resized = cv2.resize(attn_map, input_shape, interpolation=cv2.INTER_CUBIC)
    
    # 고유 ID 생성
    unique_id = uuid.uuid4()
    
    # 파일 이름에 unique_id 추가
    save_path = os.path.join(save_dir, f"{layer_name}_block_{block_idx}_sample_{sample_idx}_step_{step}.png")

    # Plot the attention map
    plt.figure(figsize=(10, 10))
    plt.imshow(attn_map_resized, cmap='viridis')
    plt.colorbar()
    plt.title(f"Total Attention Map {layer_name} - Block {block_idx} - Sample {sample_idx} - Step {step}")
    plt.savefig(save_path)
    plt.close()
