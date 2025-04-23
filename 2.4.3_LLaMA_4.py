class Llama4TextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Llama4TextAttention(config, layer_idx)
        self.use_chunked_attention = int((layer_idx + 1) % 4 != 0)  # <=> use rope
        self.is_moe_layer = layer_idx in config.moe_layers
        if self.is_moe_layer:  # the 128E model interleaves dense / sparse
            self.feed_forward = Llama4TextMoe(config)
        else:
            self.feed_forward = Llama4TextMLP(config, intermediate_size=config.intermediate_size_mlp)

        self.input_layernorm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Llama4TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_idx = layer_idx


class Llama4TextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = config.intermediate_size

        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.activation_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(down_proj)


class Llama4ForConditionalGeneration(Llama4PreTrainedModel, GenerationMixin):
    ...

    def __init__(self, config: Llama4Config):
        super().__init__(config)
        self.vision_model = Llama4VisionModel(config.vision_config)
        self.multi_modal_projector = Llama4MultiModalProjector(config)
        self.language_model = Llama4ForCausalLM(config.text_config)
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.post_init()


class Llama4MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=False,
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        return hidden_states
