...
FLASH_ATTN_FLAG=True
print("inference by Flash attention src:", FLASH_ATTN_FLAG)
...

class CoreAttention(torch.nn.Module):
...
    def forward(self, query_layer, key_layer, value_layer, attention_mask):
            pytorch_major_version = int(torch.__version__.split('.')[0])

            if pytorch_major_version >= 2:

                if FLASH_ATTN_FLAG:
                    from flash_attn import flash_attn_qkvpacked_func,flash_attn_func
                    query_layer, key_layer, value_layer = [k.permute(1, 0, 2, 3) for k in [query_layer, key_layer, value_layer]]
                    dropout_p=0.0
                    softmax_scale=0.0                    
                    context_layer = flash_attn_func(query_layer, key_layer, value_layer, dropout_p, causal=True)
                    context_layer = context_layer.permute(1, 0, 2, 3)
                #chatglm2-6b Official code
                else:
                    query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]

                    if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                        context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                        is_causal=True)
                    else:
                        if attention_mask is not None:
                            attention_mask = ~attention_mask
                        context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                        attention_mask)
                    context_layer = context_layer.permute(2, 0, 1, 3)
                new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
                context_layer = context_layer.reshape(*new_context_layer_shape)

...