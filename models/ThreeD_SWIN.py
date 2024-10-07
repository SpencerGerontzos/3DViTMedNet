import torch
import torch.nn as nn
#from timm.models.swin_transformer_V2 import SwinTransformerV2
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
# import torchvision.models.video.swin3d_t as swin3dt

class PatchEmbedding3D(nn.Module):
    def __init__(self, patch_size=2, in_channels=1, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x
    

class PatchMerging3D(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)
    
    def forward(self, x):
        """
        x: B, D*H*W, C
        """
        B, L, C = x.shape
        D, H, W = self.input_resolution
        assert L == D * H * W, "Input feature has wrong size"
        
        x = x.view(B, D, H, W, C)
        
        # Pad input if dimensions are odd
        pad_input = False
        if D % 2 == 1 or H % 2 == 1 or W % 2 == 1:
            pad_input = True
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))
        
        D, H, W = x.shape[1], x.shape[2], x.shape[3]
        
        # Merge patches
        x0 = x[:, 0::2, 0::2, 0::2, :]  # Even indices
        x1 = x[:, 1::2, 0::2, 0::2, :]  # Odd indices
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # Concatenate along channel dimension
        x = x.view(B, -1, 8 * C)  # Flatten spatial dimensions
        
        x = self.norm(x)
        x = self.reduction(x)  # Reduce dimension
        return x



class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (D_w, H_w, W_w)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # [3, B_, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * (self.head_dim ** -0.5)
        attn = (q @ k.transpose(-2, -1))  # [B_, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # [B_, N, C]
        x = self.proj_drop(self.proj(x))
        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=(0, 0, 0)):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (D, H, W)
        self.num_heads = num_heads
        self.window_size = window_size  # (D_w, H_w, W_w)
        self.shift_size = shift_size  # (D_s, H_s, W_s)
        self.mlp_ratio = 4.0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        B, L, C = x.shape
        D, H, W = self.input_resolution
        assert L == D * H * W, "Input feature has wrong size"

        x = x.view(B, D, H, W, C)

        # Cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(
                x, shifts=(-s for s in self.shift_size), dims=(1, 2, 3)
            )
        else:
            shifted_x = x

        # Partition windows
        x_windows, (pad_d, pad_h, pad_w) = self.window_partition(shifted_x)

        # W-MSA/SW-MSA
        attn_windows = self.attn(self.norm1(x_windows))

        # Merge windows
        shifted_x = self.window_reverse(attn_windows, D, H, W, pad_d, pad_h, pad_w)

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                shifted_x, shifts=self.shift_size, dims=(1, 2, 3)
            )
        else:
            x = shifted_x

        x = x.view(B, D * H * W, C)

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


    def window_partition(self, x):
        B, D, H, W, C = x.shape
        D_w, H_w, W_w = self.window_size
        #print(f"Before padding: D={D}, H={H}, W={W}")

        # Calculate padding sizes
        pad_d = (D_w - D % D_w) % D_w
        pad_h = (H_w - H % H_w) % H_w
        pad_w = (W_w - W % W_w) % W_w

        # Pad the input if necessary
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))

        # Update dimensions after padding
        D_padded, H_padded, W_padded = D + pad_d, H + pad_h, W + pad_w

        # Reshape into windows
        x = x.view(
            B,
            D_padded // D_w, D_w,
            H_padded // H_w, H_w,
            W_padded // W_w, W_w,
            C
        )
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        windows = x.view(-1, D_w * H_w * W_w, C)
        return windows, (pad_d, pad_h, pad_w)




    def window_reverse(self, windows, D, H, W, pad_d, pad_h, pad_w):
        B = int(windows.shape[0] / ((D + pad_d) * (H + pad_h) * (W + pad_w) / (self.window_size[0] * self.window_size[1] * self.window_size[2])))
        D_w, H_w, W_w = self.window_size

        D_padded, H_padded, W_padded = D + pad_d, H + pad_h, W + pad_w

        x = windows.view(
            B,
            D_padded // D_w, H_padded // H_w, W_padded // W_w,
            D_w, H_w, W_w, -1
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, D_padded, H_padded, W_padded, -1)

        # Remove padding
        x = x[:, :D, :H, :W, :].contiguous()

        return x
        



class SwinTransformer3D(nn.Module):
    def __init__(self, img_size=(28, 28, 28), patch_size=2, in_channels=1, num_classes=10, embed_dim=96, depths=[2, 2], num_heads=[3, 6]):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # Compute initial input resolution
        self.patch_embed = PatchEmbedding3D(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        patch_grid = [img_size[i] // patch_size for i in range(len(img_size))]
        self.patches_resolution = patch_grid  # [D, H, W]
        
        # Positional dropout
        self.pos_drop = nn.Dropout(0.1)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_blocks = nn.ModuleList()
            dim = int(embed_dim * 2 ** i_layer)
            input_resolution = [
                self.patches_resolution[0] // (2 ** i_layer),
                self.patches_resolution[1] // (2 ** i_layer),
                self.patches_resolution[2] // (2 ** i_layer)
            ]
            for _ in range(depths[i_layer]):
                block = SwinTransformerBlock3D(
                    dim=dim,
                    input_resolution=tuple(input_resolution),
                    num_heads=num_heads[i_layer],
                    window_size=(2, 2, 2),
                    shift_size=(0, 0, 0)
                )
                layer_blocks.append(block)
            if i_layer < self.num_layers - 1:
                # Add patch merging layer between stages
                layer_blocks.append(PatchMerging3D(tuple(input_resolution), dim))
            self.layers.append(layer_blocks)
        
        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes)


    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            for block in layer:
                x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)
        return x






# # class SwinTransformer2D(nn.Module):
# #     def __init__(self, input_size=(28, 28), num_classes=2):
# #         super(SwinTransformer2D, self).__init__()
# #         # Use 2D Swin Transformer from timm
# #         self.swin_transformer = SwinTransformerV2(
# #             img_size=input_size,
# #             patch_size=4,
# #             in_chans=1,  # Grayscale images (1 channel)
# #             embed_dim=96,
# #             depths=(2, 2, 6, 2),
# #             num_heads=(3, 6, 12, 24),
# #             window_size = 7,
# #             num_classes=num_classes
# #         )

# #     def forward(self, x):
# #         # Input x: (batch_size, channels, height, width)
# #         return self.swin_transformer(x)


# class SwinTransformer3D(nn.Module): 
#     def __init__(self, num_slices=28, num_classes=2):
#         super(SwinTransformer3D, self).__init__()
#         self.num_slices = num_slices
#         self.num_classes = num_classes
#         self.slice_classifier = SwinTransformer2D(input_size=(28, 28), num_classes=num_classes)

#     def forward(self, x, dim=2):
#         """
#         x: Input tensor of shape (batch_size, channels, depth, height, width)
#         dim: Dimension along which to take slices (2: axial, 3: coronal, 4: sagittal)
#         """
#         batch_size, channels, depth, height, width = x.shape
#         slice_preds = []

#         # Take slices along the specified dimension (axial, coronal, sagittal)
#         if dim == 2:  # Axial slices (D x H x W -> H x W)
#             slices = [x[:, :, i, :, :] for i in range(depth)]  # Extract 2D axial slices
#         elif dim == 3:  # Coronal slices (D x H x W -> D x W)
#             slices = [x[:, :, :, i, :] for i in range(height)]  # Extract 2D coronal slices
#         else:  # Sagittal slices (D x H x W -> D x H)
#             slices = [x[:, :, :, :, i] for i in range(width)]  # Extract 2D sagittal slices

#         # Classify each slice
#         for slice in slices:
#             print(slice.shape)
#             slice_pred = self.slice_classifier(slice)  # Classify the slice using 2D Swin
#             print(slice_pred)
#             slice_preds.append(slice_pred)

#         # Stack predictions and average across slices
#         slice_preds = torch.stack(slice_preds, dim=0)  # (num_slices, batch_size, num_classes)
#         slice_preds = F.softmax(slice_preds, dim=-1)  # Softmax over class scores
#         slice_preds = torch.mean(slice_preds, dim=0)  # Average predictions across slices

#         return slice_preds
