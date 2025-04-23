# use temperature to decrease the sensitivity to low probability candidates
sample_output = model.generate(input_ids, do_sample=True, max_length=50, top_k=0, temperature=0.7)

print("Output:\n" + 100 * "-")
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
