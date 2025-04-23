import torch
import torch.nn as nn
import torch.nn.functional as F
import match


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * self.heads == self.embed_size, "embed_dim need to be diveded by heads"

        self.k = nn.Liner(self.embed_size, self.embed_size, bias=False)
        self.q = nn.Liner(self.embed_size, self.embed_size, bias=False)
        self.v = nn.Liner(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Liner(self.heads * self.head_dim, self.embed_size)

    def forward(self, value, key, query, mask):
        b = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        q, k, v = self.q(query), self.k(key), self.v(value)

        # Split the embedding into self.heads different pieces
        q = q.reshape(b, query_len, self.heads, self.head_dim)
        k = k.reshape(b, key, self.heads, self.head_dim)
        v = v.reshape(b, value_len, self.heads, self.head_dim)

        # Einsum does matrix multiplication for query*keys for each training example
        # with every other key, then scale, mask, and apply softmax
        attention = torch.einsum("bqhd,bkhd->bhqk", [q, k])
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))
        attention = F.softmax(attention / mathsqrt(self.head_dim), dim=3)
        out = torch.einsum("bhqk,bkhd->bqhd", [attention, v]).reshape(b, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, inner_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(nn.Linear(embed_size, inner_dim), nn.ReLU(), nn.Linear(inner_dim, embed_size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q, mask):
        attention = self.attention(v, k, q, mask)

        # Add skip connection, followed by laver normalization
        x = self.dropout(self.norm1(attention + q))
        forward = self.feed_forward(x)

        # Add skip connection, followed by layer normalization
        out = self.dropout(self.norm2(forward + x))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len, device):
        super(PositionalEmbedding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size).to(device)
        self.encoding.requires_grad = False  # We do not want to update this during training

        pos = torch.arange(0, max_len).unsqueeze(1).float().to(device)
        _2i = torch.arange(0, embed_size, step=2).float().to(device)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_size)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_size)))

    def forward(self, x):
        # Add positional encoding to the input embeddings
        seq_len = x.shape[1]
        x = x + self.encoding[:seq_len, :]
        return x

class Encoder(nn.Module):
    def __init__(self,
            vocal_size,
            embed_size,
            num_layers,
            heads,
            device,
            inner_dim,
            drop_out,
            max_len):
    super(Encoder,self).__init__()
    self.embed_size = embed_size
    self.device = device
    self.embedding = nn.Embedding(vocal_size,embed_size)
    self.positional_encoding = PositionalEncoding(embed_size,amx_length,device)
    self.layers = nn.ModuleList([
      TransformerBlock(
        embed_size,
        heads,
        dropout = dropout,
        inner_dim=inner_dim
      ) for _ in range(num_layers)
    ])
    self.dropout = nn.Dropout(dropout)
    
  def forward(self,x,mask):
    out = self.dropout(self.word_embedding(x))
    out = self.positional_encoding(out)
    
    for layer in self.layers:
      out = layer(out,out,out,mask)
    return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size,heads,inner_dim,dropout,device):
        super(DecoderBlock,self).__init__()
        self.attention = MultiHeadAttention(embed_size,heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
                embed_size,heads,dropout,inner_dim
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x,value,key,src_mask,trg_mask):
        attention = self.atrtention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(value,key,query,src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self,
                vocal_size,
                embed_size,
                num_layers,
                heads,
                device,
                inner_dim,
                drop_out,
                max_len):
        super(Decoder,self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocal_size,embed_size)
        self.positional_encoding = PositionalEncoding(embed_size,amx_length,device)
        self.layers = nn.ModuleList([
        DecoderBlock(
            embed_size,
            heads,
            dropout = dropout,
            inner_dim=inner_dim
        ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size,vocal_size)
        
    def forward(self,x,enc_out,src_mask,trg_mask):
        x = self.dropout(self.word_embedding(x))
        x = self.positional_encoding(x)
        
        for layer in self.layers:
        x = layer(x,enc_out,enc_out,src_mask,trg_mask)
        
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self,
                src_vocab_size,
                trg_vocab_size,
                src_pad_idx,
                trg_pad_idx,
                embed_size=256,
                num_1ayers=6,
                inner_dim=1024,
                heads=8,
                dropout=0,
                device="cuda",
                max_length=100):
        super(Transformer, self).__init_()
        self.encoder = Encoder(src_vocab_size,
                            embed_size,
                            num_layers,
                            heads,
                            device,
                            inner_dim,
                            dropout,
                            max_1ength)
        self.decoder = Decoder(trg_vocab_size,
                            embed _size,
                            num_1ayers,
                            heads,
                            forward_expansion,
                            dropout,
                            device,
                            max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    def make_src_mask(self,src):
        src_mask = (src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    def make_trg mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out 
    
    
# Initialize a transformer 
modeldevice = torch.device("cuda" if torch.cuda. is_available() else "cpu")
src_vocab_size = 10000
trg_vocab_size = 10000
src_pad_idx = 0
trg_pad_idx =0
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx) .to(device)
# prepare some sample data (batch size = 1, sequence length = 10)
src = torch.tensor([[1, 5, 6, 2, 0, 0, 0, 0, 0, 0]], device=device)
trg = torch.tensor([[1, 7, 4, 3, 2, 0, 0, 0, 0, 0]], device=device)
# perform a forward pass
out = model(src, trg[:, :-1])
print(out.shape) # Expected shape: (batch_size, trg_seg_length - 1, trg vocab_ size)