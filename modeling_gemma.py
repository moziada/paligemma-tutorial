import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel
from torch.nn import functional as F

class KVCache():
    def __init__(self) -> None:
        # each cached list probably contains N elements for N layer idxs
        self.key_cache: List[torch.Tensor]=[]
        self.value_cache: List[torch.Tensor]=[]
        
    def num_items(self) -> int:
        if len(self.key_cache)==0:
            return 0
        else:
            # shape of key_cache: (batch_size, n_heads, seq_len, hidden_size)
            return self.key_cache[0].shape[-2]
        
    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # create initial caches for that layer
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # concat new cachs for that layer to already existing ones along seq_len dim
            self.key_cache[layer_idx] = torch.cat(self.key_cache[layer_idx], key_states, dim=-2)
            self.value_cache[layer_idx] = torch.cat(self.value_cache[layer_idx], value_states, dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
        
class GemmaConfig():
    def __init__(
            self,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,    # query heads (query heads are different than kv heads, search for grouped query attention)
            num_key_value_heads,
            head_dim=256,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            attention_bias=False,
            attention_dropout=0.0,
            pad_token_id=None,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():
    def __init__(
            self,
            vision_config=None,
            text_config=None,
            ignore_index=-100,
            image_token_index=256000,
            vocab_size=257152,
            projection_dim=2048,
            hidden_size=2048,
            pad_token_id=None,
            **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        y = self.gate_proj(x)   # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, intermediate_size)
        y = nn.functional.gelu(y, approximate="tanh")
        
        x_proj = self.up_proj(x)    # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, intermediate_size)
        z = y * x_proj  # (batch_size, seq_len, intermediate_size)
        return self.down_proj(z)    # (batch_size, seq_len, hidden_size)

def repeat_kv(hidden_state: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, n_kv_heads, q_len, hidden_size = hidden_state.shape
    if n_rep == 1:
        return hidden_state
    hidden_state = hidden_state[:, :, None, :, :].expand(batch_size, n_kv_heads, n_rep, q_len, hidden_size)
    return hidden_size.reshape(batch_size, n_kv_heads*n_rep, q_len, hidden_size)

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: (batch_size, num_heads, seq_len, head_dim)
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)   # -> (batch_size, head_dim // 2, 1)
        position_ids_expanded = position_ids[:, None, :].float()    # -> (batch_size, 1, seq_len)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)     # -> (batch_size, head_dim // 2, seq_len).transpose(1, 2) --> (batch_size, seq_len, head_dim // 2)
            emb = torch.cat((freqs, freqs), dim=-1)     # -> (batch_size, seq_len, head_dim)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]  # first half of last dim
    x2 = x[..., x.shape[-1] // 2:]  # second half of last dim
    return torch.cat((x1, x2), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)  # add head dimension
    sin = sin.unsqueeze(unsqueeze_dim)  # add head dimension

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int]=None):
        super().__init__()
        self.config = config
        self.layer_idx=layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0

        # n_heads: 8
        # num_key_value_heads: 1
        # hidden_size: 1024
        # head_dim: 1024 / 8 = 128
        # Wq: (1024, 8*128) -> (1024, 1024)
        # Wk: (1024, 1*128) -> (1024, 128)
        # Wv: (1024, 1*128) -> (1024, 128)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotery_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor]=None,
            postion_ids: Optional[torch.LongTensor]=None,
            kv_cache: Optional[KVCache]=None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, q_len, _ = hidden_states.size()

        query_state = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_state = query_state.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        print(postion_ids)
        cos, sin = self.rotery_emb(value_states, postion_ids, seq_len=None)
        query_state, key_states = apply_rotary_pos_emb(query_state, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # this repeates heads of k and v to match q num of heads (this cancels the grouped query attention optimization)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_state, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # -> (batch_size, num_heads_q, seq_len_q, seq_len_kv)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_state.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states) # -> (batch_size, num_heads_q, seq_len_q, head_dim)
        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of shape: {(batch_size, self.num_heads, q_len, self.head_dim)}, but got "
                f"{(attn_output.size())}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()  # make seq_len the second dim
        attn_output = attn_output.view(batch_size, q_len, -1)   # concat heads

        # Mix the independent heads representation
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)     # (batch_size, seq_len, hidden_size)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        return hidden_states
    
class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        
        hidden_states = inputs_embeds   # (batch_size, seq_len, hidden_size)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states    # (batch_size, seq_len, hidden_size)

class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
            self,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ):
        # inputs_embeds: (batch_size, seq_len, hidden_size)
        # outputs_embeds: (batch_size, seq_len, hidden_size)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        logits = self.lm_head(outputs)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }
        if kv_cache is not None:
            return_data['kv_cache'] = kv_cache
        
        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # (batch_size, num_patchs, embed_dim) -> (batch_size, num_patchs, projection_dim)
        hidden_states = self.linear(image_features)
        return hidden_states
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        self.language_model = GemmaForCausalLM(config.text_config)

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
            self,
            image_embeds: torch.Tensor,
            inputs_embeds: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            kv_cache: Optional[KVCache]=None,
    ):
        _, _, embed_dim = image_embeds.shape
        batch_size, seq_len = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        scaled_image_embeds = image_embeds / self.config.hidden_size**0.5

        combined_embeds = torch.zeros(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
        image_mask = input_ids == self.config.image_token_index
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        pad_mask = input_ids == self.config.pad_token_id
        # mask shapes -> (batch_size, seq_len)

        text_mask_broadcasted = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_broadcasted = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_broadcasted = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        # unsqueeze(-1) -> (batch_size, seq_len, 1)
        # expand(-1, -1, embed_dim) -> (batch_size, seq_len, embed_dim) (broadcasted)

        combined_embeds = torch.where(text_mask_broadcasted, inputs_embeds, combined_embeds)     # Copies text embeddings
        combined_embeds = combined_embeds.masked_scatter(image_mask_broadcasted, scaled_image_embeds)       # Copies image embeddings
        combined_embeds = torch.where(pad_mask_broadcasted, torch.zeros_like(combined_embeds), combined_embeds)     # Putting zero for padding

        ### ATTENTION MASK ###

        if kv_cache is None or kv_cache.num_items()==0:
            # Prefilling phase
            causual_mask = torch.full((batch_size, seq_len, seq_len), fill_value=0, dtype=dtype, device=device)
        else:
            # Token generation phase
            assert seq_len == 1
            kv_len = kv_cache.num_items() + 1
            causual_mask = torch.full((batch_size, seq_len, kv_len), fill_value=0, dtype=dtype, device=device)
        
        # Extend head dim (batch_size, seq_len, kv_len) -> (batch_size, Num_heads, seq_len, kv_len)
        causual_mask = causual_mask.unsqueeze(1)

        ### POSITIONAL ENCODING ###
        if kv_cache is not None and kv_cache.num_items() > 0:
            # Prefilling phase, add position ids [0, 1, ..., N] where N is equal to num of images patches + prompt length
            position_ids = attention_mask.cumsum(-1)[:, -12]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)
        
        return combined_embeds, causual_mask, position_ids

    def forward(
            self,
            input_ids: torch.LongTensor=None,
            pixel_values: torch.FloatTensor=None,
            attention_mask: Optional[torch.Tensor]=None,
            kv_cache: Optional[KVCache]=None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "Input cannot be padded"

        # 1. Pass <image> + user_prompt tokens
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)    # (batch_size, seq_length, hidden_size)

        # 2. Get actual image embeddings
        image_embeds = self.vision_tower(pixel_values.to(inputs_embeds.dtype))   # (batch_size, n_patchs, embed_dim)

        projected_image_embeds = self.multi_modal_projector(image_embeds)   # (batch_size, n_patchs, hidden_size)

        # 3. replace actual image embeddings
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            projected_image_embeds,
            inputs_embeds,
            input_ids,
            attention_mask,
            kv_cache,
        )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs