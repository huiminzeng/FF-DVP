import torch
from torch import nn

class VisionAdapter(nn.Module):
    def __init__(self, vision_encoder, patch_size, prompt_dim, num_layers, num_tokens, hidden_size):
        super().__init__()
        # initiate prompt:
        # val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -5, 5)

        # deep prompt
        self.deep_prompt_embeddings = nn.Parameter(torch.zeros(int(num_layers//2), num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.deep_prompt_embeddings.data, -5, 5)

        # project prompt embeddings
        self.prompt_proj = nn.Linear(prompt_dim, hidden_size)
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')
        
        self.prompt_dropout = nn.Dropout(0.5)

        # CLIP visual encoder
        self.encoder = vision_encoder

        # register hyperparameters
        self.num_tokens = num_tokens

    def incorporate_prompt(self, x):
        # first layer
        B = x.shape[0]
        # after CLS token, all before image patches
        # this is taken from clip_model.py
        x = self.encoder.conv1(x)  # (*, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1) # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1) # shape = [* grid**2, width]
        x = torch.cat([self.encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # combine prompt embeddings with image-patch embeddings
        x = torch.cat((
                x[:1, :, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)).permute(1, 0, 2),
                x[1:, :, :]
            ), dim=0)
        # (1 + n_prompt + (n_patches-1), batch_size, hidden_dim)
        return x
    
    def forward_deep_prompt(self, x):
        attn_weights = []
        hidden_states = None
        weights = None

        B = x.shape[1]
        num_layers = len(self.encoder.transformer.resblocks)
        
        for i in range(num_layers):
            if i == 0:
                hidden_states = self.encoder.transformer.resblocks[i](x)

            elif i <= int(num_layers // 2):
                deep_prompt_emb = self.prompt_dropout(self.prompt_proj(self.deep_prompt_embeddings[i-1]).expand(B, -1, -1)).permute(1, 0, 2)
                hidden_states = torch.cat((
                    hidden_states[:1, :, :],
                    deep_prompt_emb,
                    hidden_states[(1+self.num_tokens):, :, :]
                ), dim=0)

                hidden_states = self.encoder.transformer.resblocks[i](hidden_states)
                
            else:
                hidden_states = self.encoder.transformer.resblocks[i](hidden_states)

        return hidden_states

    def forward(self, x):
        
        x = self.incorporate_prompt(x)
        x = self.forward_deep_prompt(x)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.encoder.ln_post(x[:, 0, :])

        if self.encoder.proj is not None:
            x = x @ self.encoder.proj

        return x