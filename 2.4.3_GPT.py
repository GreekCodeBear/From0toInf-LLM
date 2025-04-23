class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        ...

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual


##################################################################

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"  # small version
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt = "Once upon a time,"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

model.eval()
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50, num_beams=1, do_sample=True, temperature=0.7)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)

##################################################################

from datasets import load_dataset
from transformers import Trainer, TrainingArguments

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# example: fine-tune on WikiText2 language modeling


def encode(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


encoded_dataset = dataset.map(encode, batched=True)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    logging_steps=100,
)


def data_collator(features):
    # collate into batch
    return {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
        "labels": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    data_collator=data_collator,
)

trainer.train()
