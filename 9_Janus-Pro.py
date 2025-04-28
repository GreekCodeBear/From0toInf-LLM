class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(params.n_embed, params.image_token_embed)
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(params.image_token_embed, params.image_token_size)

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x
