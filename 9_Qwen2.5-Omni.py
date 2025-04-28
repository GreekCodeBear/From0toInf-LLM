class Qwen2_5OmniThinkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    def __init__(self, config: Qwen2_5OmniThinkerConfig):
        super().__init__(config)
        # 音频与视觉编码模块（用于多模态融合）
        self.audio_tower = Qwen2_5OmniAudioEncoder._from_config(config.audio_config, attn_implementation=config._attn_implementation)
        self.visual = Qwen2_5OmniVisionEncoder._from_config(config.vision_config, attn_implementation=config._attn_implementation)
        # 文本部分的 transformer 模型
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen2_5OmniThinkerModel._from_config(config.text_config, attn_implementation=config._attn_implementation)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Qwen2_5OmniThinkerCausalLMOutputWithPast:
        # === 1. 输入嵌入与多模态融合 ===
        if inputs_embeds is None:
            # 从文本 token 得到嵌入
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # 保存一份原始嵌入，用于后续 Talker 模块作为条件信息
            embeds_to_talker = inputs_embeds.clone()

            # 如果存在音频特征，则融合音频编码
            if input_ids.shape[1] != 1:
                if input_features is not None:
                    audio_outputs = self.audio_tower(input_features, feature_lens=..., aftercnn_lens=...)
                    audio_features = audio_outputs.last_hidden_state
                    audio_mask = (input_ids == self.config.audio_token_index).unsqueeze(-1)
                    inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
                    embeds_to_talker = embeds_to_talker.masked_scatter(audio_mask, torch.zeros_like(audio_features))
                # 同理处理图像与视频输入...
        else:
            # 如果已经提供了嵌入，则直接使用
            embeds_to_talker = inputs_embeds.clone()

        # === 2. 构造位置 ids 和因果注意力 mask ===
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                # 根据输入及多模态信息计算专用的 3D 位置 id（比如文本与视觉位置不同）
                position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask, ..., audio_feature_lengths, ...)
            else:
                # 针对缓存续写情况，按 offset 生成 position_ids
                position_ids = generate_position_ids_from_cache(input_ids, cache_position, rope_deltas)
        # === 3. 调用文本生成 Transformer ===
        outputs = self.model(attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # === 4. 输出时保留两份信息 ===
        # 第一份：原始用于 Talker 模块条件的嵌入（embeds_to_talker）
        # 第二份：完整的 Transformer 隐藏状态
        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(embeds_to_talker, outputs.hidden_states),
            attentions=outputs.attentions,
            attention_mask=attention_mask,
            rope_deltas=rope_deltas,
        )


###################################################################


class Qwen2_5OmniTalkerForConditionalGeneration(Qwen2_5OmniPreTrainedModelForConditionalGeneration, GenerationMixin):
    def __init__(self, config: Qwen2_5OmniTalkerConfig):
        super().__init__(config)
        # 用于将 Thinker 模块输出的 embedding 投影到 Talker 模型的隐层维度
        self.thinker_to_talker_proj = nn.Linear(config.embedding_size, config.hidden_size)
        # Talker 模型本体（内部实现了 Transformer 解码器）
        self.model = Qwen2_5OmniTalkerModel(config)
        # 将 Talker 模型输出映射到语音令牌（codec）词表空间
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 定义一些特殊 token（例如用于语音编码的 BOS、PAD 等）
        self.codec_bos_token = config.tts_codec_start_token_id
        self.codec_pad_token = config.tts_codec_pad_token_id
        self.text_bos_token = config.tts_text_start_token_id
        self.text_eos_token = config.tts_text_end_token_id
        self.text_pad_token = config.tts_text_pad_token_id
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Qwen2_5OmniTalkerCausalLMOutputWithPast:
        # 如果没有直接提供嵌入，则通过 token 嵌入层获取
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds = codec_embeds + thinker_reply_part[:, :1, :]
            if thinker_reply_part.shape[1] > 1:
                thinker_reply_part = thinker_reply_part[:, 1:, :]
        # 这里的 inputs_embeds 来自 Thinker 模块经过处理后，
        # 接下来我们将其通过投影层转换为 Talker 模型所需的维度：
        talker_inputs_embeds = self.thinker_to_talker_proj(inputs_embeds)

        # 传入的 position_ids、attention_mask、cache_position 等均用于生成因果 mask 与位置编码
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=talker_inputs_embeds,
            use_cache=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        # 将 Talker 模型的隐藏状态通过 codec_head 转换为语音令牌的 logits
        logits = self.codec_head(hidden_states)
        logits = logits.float()

        # 返回的输出包含 logits、past_key_values、以及可能的其它信息（例如隐藏状态、注意力等）
        return Qwen2_5OmniTalkerCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            attention_mask=attention_mask,
            rope_deltas=kwargs.get("rope_deltas", None),
            thinker_reply_part=thinker_reply_part,
        )


#####################################################################

# 1. Generate from thinker module
thinker_result = self.thinker.generate(
    input_ids=input_ids,
    return_dict_in_generate=True,
    output_hidden_states=True,
    **thinker_kwargs,
)
if not (return_audio and self.has_talker):
    return thinker_result.sequences

# 2. Generate speech tokens from talker module
thinker_generate_ids = thinker_result.sequences[:, input_ids.size(1) :].to(self.talker.device)
thinker_token_embeds = [x[0].to(self.talker.device) for x in thinker_result.hidden_states]
thinker_hidden_states = [x[1][-1].to(self.talker.device) for x in thinker_result.hidden_states]

talker_text_bos_token = speaker_params["bos_token"]
talker_input_text_ids = torch.cat(
    [
        input_ids.to(self.talker.device),
        torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.talker.device),
        thinker_generate_ids[:, :1],
    ],
    dim=-1,
)

talker_input_ids = torch.cat(
    [
        torch.full_like(input_ids, fill_value=self.talker.codec_mask_token, device=self.talker.device),
        torch.tensor([[self.talker.codec_pad_token]], dtype=torch.long, device=self.talker.device),
        torch.tensor([[self.talker.codec_bos_token]], dtype=torch.long, device=self.talker.device),
    ],
    dim=1,
)

thinker_reply_part = torch.cat(thinker_hidden_states[1:], dim=1) + torch.cat(thinker_token_embeds[1:], dim=1)
talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
talker_inputs_embeds = torch.cat(
    [
        talker_inputs_embeds,
        self.thinker.get_input_embeddings()(torch.tensor([[talker_text_bos_token]], dtype=torch.long, device=self.thinker.device)).to(self.talker.device),
        thinker_reply_part[:, :1, :],
    ],
    dim=1,
)

thinker_reply_part = torch.cat(
    [
        thinker_reply_part[:, 1:, :],
        self.thinker.get_input_embeddings()(torch.tensor([[self.talker.text_eos_token]], dtype=torch.long, device=self.thinker.device)).to(self.talker.device),
        self.thinker.get_input_embeddings()(torch.tensor([[self.talker.text_pad_token]], dtype=torch.long, device=self.thinker.device)).to(self.talker.device),
    ],
    dim=1,
)

talker_attention_mask = torch.cat([kwargs["attention_mask"], kwargs["attention_mask"].new_ones((1, 2))], dim=1).to(self.talker.device)

talker_result = self.talker.generate(
    input_ids=talker_input_ids,
    input_text_ids=talker_input_text_ids,
    thinker_reply_part=thinker_reply_part,
    inputs_embeds=talker_inputs_embeds,
    attention_mask=talker_attention_mask,
    suppress_tokens=[self.talker.codec_bos_token],
    **{k: (v.to(self.talker.device) if torch.is_tensor(v) else v) for k, v in talker_kwargs.items()},
)

#####################################################################


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x
