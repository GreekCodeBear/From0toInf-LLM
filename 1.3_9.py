# set seed to reproduce results. Feel free to change the seed though to get different results
tf.random.set_seed(0)

# deactivate top_k sampling and sample only from 92% most likely words
sample_output = model.generate(input_ids, do_sample=True, max_length=50, top_p=0.92, top_k=0)

print("Output:\n" + 100 * "-")
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
