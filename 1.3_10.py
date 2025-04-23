import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# add the EOS token as PAD token to avoid warnings
model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)
# encode context the generation is conditioned on
input_ids = tokenizer.encode("I enjoy walking with my cute dog", return_tensors="tf")

# activate beam search and early_stopping
beam_output = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

print("Output:\n" + 100 * "-")
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))
