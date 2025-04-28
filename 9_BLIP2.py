class BertLayer(nn.Module):
    ...
    def forward(...):
        ...
        # 如果有 query_length，需要对 query tokens 和剩余 tokens 分开处理
        if query_length > 0:
            # 取出前 query_length 个 token 的注意力输出
            query_attention_output = attention_output[:, :query_length, :]
            # 如果存在 cross-attention，就只对 query 部分进行跨注意力
            if self.has_cross_attention:
                ...
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                # cross_attention_outputs[0]: 跨注意力后的 query 部分
                query_attention_output = cross_attention_outputs[0]
            # 对 query 部分做专门的前馈网络 (feed_forward_chunk_query)
            layer_output_query = apply_chunking_to_forward(
                self.feed_forward_chunk_query,  # 针对 query 的前馈网络
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )
        ...
        # 将最终输出放回 outputs，并加上 present_key_value
        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)
        return outputs

###########################################################################
    
@registry.register_model("blip2_opt")
class Blip2OPT(Blip2Base):
    def __init__(
        ...
    ):
        super().__init__()
        # === 1. 初始化分词器 ===
        self.tokenizer = self.init_tokenizer()

        # === 2. 初始化视觉编码器并视需要冻结 ===
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model,
            img_size,
            drop_path_rate,
            use_grad_checkpoint,
            vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # 3. 初始化 Qformer：不对Qformer做requires_grad=False，所以它是可训练的 
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            # 这里是去除文本部分 BERT 的 FFN
            layer.output = None
            layer.intermediate = None

        # === 4. 初始化并冻结 OPT 模型 ===
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False

        # === 5. 投影层 ===
        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        
        
class Blip2Qformer(Blip2Base):
    def forward(self, samples):
        image = samples["image"]
        text = samples["text_input"]

        # === 视觉编码 ===
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        image_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)

        # === 文本编码 ===
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        # ------------------- 1. Image-Text Contrastive (ITC) Loss --------
        image_feats_all = concat_all_gather(image_feats)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)      # [batch_size*num_gpu, embed_dim]
        
        # 相似度计算，聚合 query tokens
        sim_q2t = torch.matmul(image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)).squeeze()
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1),
            image_feats_all.permute(0, 2, 1)
        ).squeeze()
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp

        # 目标索引（正样本）
        rank = dist.get_rank()
        bs = image.size(0)
        targets = torch.linspace(
            rank * bs, rank * bs + bs - 1, bs, dtype=int
        ).to(image.device)

        # 如果有 image_id，就用特殊的 soft label；否则用 cross_entropy
        if "image_id" in samples.keys():
            ...
            loss_itc = (loss_t2i + loss_i2t) / 2
        else:
            loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) / 2

        # ------------------- 2. Image-Text Matching (ITM) Loss -----------
        ...
        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            ...
            encoder_hidden_states=image_embeds_all,
            ...
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        # ------------------- 3. Language Modeling (LM) Loss -------------
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=torch.cat(
                [
                    torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device),
                    text_tokens.attention_mask
                ],
                dim=1
            ),
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )
        loss_lm = lm_output.loss

        # ------------------- Total Loss -------------------
        return BlipOutput(
            loss=loss_itc + loss_itm + loss_lm,  # 总损失
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )


##########################################################################


itm_labels = torch.cat(
    [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
    dim=0,
).to(image.device)

##########################################################################

class BertLMHeadModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()
        ......
        

########################################################################

class BertLayer(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.layer_num = layer_num
        if (
            self.config.add_cross_attention
            and layer_num % self.config.cross_attention_freq == 0
        ):
            self.crossattention = BertAttention(
                config, is_cross_attention=self.config.add_cross_attention
            )
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)

#######################################################################

......
def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        ......