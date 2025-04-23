llm = LLM(
    model="meta-llama/Llama-2-7b-hf", enable_chunked_prefill=True
)  # Set max_num_batched_tokens to tune performance.# NOTE: 2048 is the default max_num_batched_tokens for chunked prefill.# llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_chunked_prefill=True, max_num_batched_tokens=2048)
