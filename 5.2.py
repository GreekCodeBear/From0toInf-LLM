class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        ...
        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
        ...

    def forward(
        self,
        input_ids=None,
        attention_mask=None,...):
        ...
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        last_hidden_state = base_model_output.hidden_states[-1]
        ...
        value = self.v_head(last_hidden_state).squeeze(-1)
        
        
class ValueHead(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        ...
        self.summary = nn.Linear(hidden_size, 1)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        output = self.summary(output)
        return output
    
    
for epoch in tqdm(range(NUM_EPOCH), "epoch: "):
    if epoch >= config.total_ppo_epochs:
        break
    for batch in tqdm(ppo_trainer.dataloader): 
        # Rollout：generate response from sampled prompt
        question_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs)
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        # Evaluation：Compute reward score (using the sentiment analysis pipeline)
        texts = ["Question: " + q + "\n\nAnswer: " + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
        # Optimization：Run PPO step
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)


def compute_rewards(
    self,
    scores: torch.FloatTensor,
    logprobs: torch.FloatTensor,
    ref_logprobs: torch.FloatTensor,
    masks: torch.LongTensor,
):
    rewards, non_score_rewards, kls = [], [], []
    for score, logprob, ref_logprob, mask in zip(scores, logprobs, ref_logprobs, masks):
        # compute KL penalty (from difference in logprobs)
        kl = self._kl_penalty(logprob, ref_logprob)
        kls.append(kl)
        non_score_reward = -self.kl_ctl.value * kl
        non_score_rewards.append(non_score_reward)
        reward = non_score_reward.clone()
        last_non_masked_index = mask.nonzero()[-1]

        # reward is preference model score + KL penalty
        reward[last_non_masked_index] += score
        rewards.append(reward)



def compute_advantages(
    self,
    values: torch.FloatTensor,
    rewards: torch.FloatTensor,
    mask: torch.FloatTensor,
):
    lastgaelam = 0
    advantages_reversed = []
    gen_len = rewards.shape[-1]

    values = values * mask
    rewards = rewards * mask

    if self.config.whiten_rewards:
        rewards = masked_whiten(rewards, mask, shift_mean=False)

    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
        lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

    returns = advantages + values
    advantages = masked_whiten(advantages, mask)
    advantages = advantages.detach()
    return values, advantages, returns



def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        ratio = torch.exp(logprobs - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss