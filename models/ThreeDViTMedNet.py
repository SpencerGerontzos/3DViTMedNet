import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
from torchsummary import summary

import matplotlib.pyplot as plt


def shifting_image_procedure(x):
    x = x.float()

    # Define shifts for 3D: (left, right, top, bottom, front, back)
    shifts = (
        (1, -1, 0, 0, 0, 0),  # Shift in depth
        (-1, 1, 0, 0, 0, 0),  # Reverse shift in depth
        (0, 0, 1, -1, 0, 0),  # Shift in height
        (0, 0, -1, 1, 0, 0),  # Reverse shift in height
        (0, 0, 0, 0, 1, -1),  # Shift in width
        (0, 0, 0, 0, -1, 1),  # Reverse shift in width
    )

    # Apply padding shifts
    shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))

    # Concatenate original and shifted versions of x
    x_with_shifts = torch.cat((x, *shifted_x), dim=1)

    #print(x_with_shifts.shape)
    return x_with_shifts



class CNN_3D_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN_3D_Layer, self).__init__()

        # Adjust in_channels based on the number of shifted images
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out


class SliceExtractorWithEmbedding(nn.Module):
    def __init__(self, input_channels: int, embedding_dim: int = 256, num_tokens: int = 84):
        super(SliceExtractorWithEmbedding, self).__init__()

        # Load pre-trained ResNet-50 and modify the input and output layers
        self.resnet_2d = models.resnet50(pretrained=True)
        self.resnet_2d.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.resnet_2d.fc = nn.Identity()

        # Non-linear projection layers
        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )

        # Embedding tokens and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.sep_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.plane_embeddings = nn.Parameter(torch.randn(3, embedding_dim))  # For coronal, sagittal, axial
        self.positional_embeddings = nn.Parameter(torch.randn(num_tokens + 4, embedding_dim))

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape

        # Extract slices along each plane
        coronal_slices = [x[:, :, i, :, :] for i in range(depth)]         # List of [B, C, H, W]
        sagittal_slices = [x[:, :, :, i, :] for i in range(height)]       # List of [B, C, D, W]
        axial_slices = [x[:, :, :, :, i] for i in range(width)]           # List of [B, C, D, H]

        # Stack slices and reshape for CNN input
        coronal_stack = torch.stack(coronal_slices, dim=1)                # [B, D, C, H, W]
        sagittal_stack = torch.stack(sagittal_slices, dim=1)              # [B, H, C, D, W]
        axial_stack = torch.stack(axial_slices, dim=1)                    # [B, W, C, D, H]

        # Reshape to (B * N, C, H, W) for CNN processing
        coronal_input = coronal_stack.view(-1, channels, height, width)
        sagittal_input = sagittal_stack.view(-1, channels, depth, width)
        axial_input = axial_stack.view(-1, channels, depth, height)

        # Pass through 2D CNN
        coronal_features = self.resnet_2d(coronal_input)    # [B * D, 2048]
        sagittal_features = self.resnet_2d(sagittal_input)  # [B * H, 2048]
        axial_features = self.resnet_2d(axial_input)        # [B * W, 2048]

        # Apply non-linear projection
        coronal_proj = self.projection(coronal_features)    # [B * D, embedding_dim]
        sagittal_proj = self.projection(sagittal_features)  # [B * H, embedding_dim]
        axial_proj = self.projection(axial_features)        # [B * W, embedding_dim]

        # Reshape back to (B, N, embedding_dim)
        coronal_tokens = coronal_proj.view(batch_size, depth, -1)
        sagittal_tokens = sagittal_proj.view(batch_size, height, -1)
        axial_tokens = axial_proj.view(batch_size, width, -1)

        # Concatenate tokens with cls and sep tokens
        tokens = torch.cat([
            self.cls_token.repeat(batch_size, 1, 1),         # [B, 1, embedding_dim]
            coronal_tokens,                                  # [B, D, embedding_dim]
            self.sep_token.repeat(batch_size, 1, 1),         # [B, 1, embedding_dim]
            sagittal_tokens,                                 # [B, H, embedding_dim]
            self.sep_token.repeat(batch_size, 1, 1),         # [B, 1, embedding_dim]
            axial_tokens,                                    # [B, W, embedding_dim]
            self.sep_token.repeat(batch_size, 1, 1)          # [B, 1, embedding_dim]
        ], dim=1)  # Expected shape: [B, total_tokens + 4, embedding_dim] = [B, 88, embedding_dim]

        # Apply plane embeddings
        tokens_without_cls_sep = tokens[:, 1:, :]  # Exclude the cls_token

        # Create plane_embeddings tensor
        plane_embeddings = torch.zeros(batch_size, tokens_without_cls_sep.size(1), tokens_without_cls_sep.size(2), device=x.device)

        # Define indices for each plane in tokens_without_cls_sep
        coronal_indices = torch.arange(0, depth)  # Positions 0 to 27
        sep1_index = depth                        # Position 28

        sagittal_start = depth + 1                # Position 29
        sagittal_indices = torch.arange(sagittal_start, sagittal_start + height)  # Positions 29 to 56
        sep2_index = sagittal_start + height      # Position 57

        axial_start = sep2_index + 1              # Position 58
        axial_indices = torch.arange(axial_start, axial_start + width)            # Positions 58 to 85
        sep3_index = axial_start + width          # Position 86

        # Assign plane embeddings
        plane_embeddings[:, coronal_indices, :] = self.plane_embeddings[0]
        plane_embeddings[:, sagittal_indices, :] = self.plane_embeddings[1]
        plane_embeddings[:, axial_indices, :] = self.plane_embeddings[2]
        # Sep tokens and cls token remain without plane embeddings

        # Add plane embeddings to tokens_without_cls_sep
        tokens_without_cls_sep += plane_embeddings

        # Reconstruct tokens with cls_token
        tokens = torch.cat([tokens[:, :1, :], tokens_without_cls_sep], dim=1)

        # Add positional embeddings
        tokens += self.positional_embeddings

        return tokens



class MSA(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout_rate: float = 0.0):
        super(MSA, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim      # Total embedding dimension
        self.num_heads = num_heads      # Number of attention heads
        self.head_dim = embed_dim // num_heads  # Dimension per head

        # Linear layers for query, key, and value projections
        self.qkv_projection = nn.Linear(embed_dim, embed_dim * 3)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.size()

        # Project input embeddings to queries, keys, and values
        qkv = self.qkv_projection(x)  # Shape: (batch_size, seq_length, embed_dim * 3)

        # Reshape and permute to separate heads
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: (3, batch_size, num_heads, seq_length, head_dim)

        # Unpack queries, keys, and values
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # Compute scaled dot-product attention scores
        scaling_factor = self.head_dim ** 0.5
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / scaling_factor

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute context vector as weighted sum of values
        context = torch.matmul(attention_weights, values)

        # Concatenate heads and reshape
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)

        # Final linear projection
        output = self.output_projection(context)

        return output

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout_rate: float = 0.0,
                 forward_expansion: int = 2,
                 forward_dropout_rate: float = 0.0,
                 **kwargs):
        super(TransformerEncoderBlock, self).__init__()

        # Layer normalization and multi-head self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MSA(embed_dim=embed_dim, num_heads=num_heads, dropout_rate=dropout_rate, **kwargs)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Feedforward network (FFN)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.GELU(),
            nn.Dropout(forward_dropout_rate),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        """
        Forward pass for the Transformer encoder block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, embed_dim)
        """
        # First residual connection with multi-head attention
        x_residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x_residual + self.dropout1(x)

        # Second residual connection with feedforward network
        x_residual = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = x_residual + x  # Residual connection (dropout included in feedforward)

        return x



class VanillaEncoder(nn.Module):
    def __init__(self, depth: int = 8, **kwargs):
        super(VanillaEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(**kwargs) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    


class ClassificationHead(nn.Module):

    def __init__(self, emb_size: int = 256, n_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        cls_token = x[:, 0]
        return self.linear(cls_token)#, attn_map
        
class ThreeDViTMedNet(nn.Sequential):
    def __init__(self,
                in_channels: int = 7,
                out_channels: int = 32,
                emb_size: int = 256,
                depth: int = 8,
                n_classes: int = 2,
                **kwargs):
        super().__init__()

        self.cnn_3d_layer = CNN_3D_Layer(in_channels, out_channels)
        self.slice_extractor_with_embedding = SliceExtractorWithEmbedding(out_channels)
        self.vanilla_encoder = VanillaEncoder(depth, **kwargs)
        self.classification_head = ClassificationHead(emb_size, n_classes)
    def forward(self, x):
        x = shifting_image_procedure(x)
        x = self.cnn_3d_layer(x)
        x = self.slice_extractor_with_embedding(x)
        x = self.vanilla_encoder(x)
        x = self.classification_head(x)
        return x











# def shifting_image_procedure(x):
#     x = x.float()

#     # Define shifts for 3D: (left, right, top, bottom, front, back)
#     shifts = (
#         (1, -1, 0, 0, 0, 0),  # Shift in depth
#         (-1, 1, 0, 0, 0, 0),  # Reverse shift in depth
#         (0, 0, 1, -1, 0, 0),  # Shift in height
#         (0, 0, -1, 1, 0, 0),  # Reverse shift in height
#         (0, 0, 0, 0, 1, -1),  # Shift in width
#         (0, 0, 0, 0, -1, 1),  # Reverse shift in width
#     )

#     # Apply padding shifts
#     shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))

#     # Concatenate original and shifted versions of x
#     x_with_shifts = torch.cat((x, *shifted_x), dim=1)

#     #print(x_with_shifts.shape)
#     return x_with_shifts



# class CNN_3D_Layer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(CNN_3D_Layer, self).__init__()

#         # Adjust in_channels based on the number of shifted images
#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
#         self.bn1 = nn.BatchNorm3d(out_channels)
#         self.relu1 = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#         self.relu2 = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         out = self.relu2(self.bn2(self.conv2(out)))
#         return out


# # class Slice_Extractor(nn.Module):
# #     def __init__(self, out_channels: int):
# #         super(Slice_Extractor, self).__init__()
# #         # 2D CNN part
# #         # Load the pre-trained ResNet-18 model and Extract the global average pooling layer
# #         self.CNN_2D = models.resnet50(weights=True)
# #         self.CNN_2D.conv1 = nn.Conv2d(out_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
# #         self.CNN_2D.fc = nn.Identity()

# #         # Non - Linear Projection block
# #         self.non_linear_proj = nn.Sequential(
# #             nn.Linear(2048, 512),
# #             nn.ReLU(),
# #             nn.Linear(512, 256)
# #             )

# #     def forward(self, input_tensor):
# #         B, C, D, H, W = input_tensor.shape
# #         # Extract coronal features
# #         coronal_slices = torch.split(input_tensor, 1, dim=2)                      # This gives us a tuple of length 128, where each element has shape (batch_size, channels, 1, width, height) 
# #         Ecor = torch.cat(coronal_slices, dim=2)                                   # lets concatenate along dimension 2 to get the desired output shape for Ecor: R^C3d×N×W×H.

# #         saggital_slices = torch.split(input_tensor.clone(), 1, dim = 3)           # This gives us a tuple of length 128, where each element has shape (batch_size, channels, length, 1, height) 
# #         Esag = torch.cat(saggital_slices, dim = 3)                                # lets concatenate along dimension 3 to get the desired output shape for Ecor: R^C3d×L×N×H.

# #         axial_slices = torch.split(input_tensor.clone(), 1, dim = 4)              # This gives us a tuple of length 128, where each element has shape (batch_size, channels, length, width, 1) 
# #         Eax = torch.cat(axial_slices, dim = 4)                                    # lets concatenate along dimension 3 to get the desired output shape for Ecor: R^C3d×L×W×N.

# #         # Lets calculate S using E for X
# #         # after matirx multiplications, we reshape the outputs based on its plane for concatenation 
# #         Scor = (Ecor * input_tensor).permute(0, 2, 1, 3, 4).contiguous()          # Scor will now have a shape (batch_size, N, channels, width, height) 
# #         Ssag = (Esag * input_tensor).permute(0, 3, 1, 2, 4).contiguous()          # Ssag will now have a shape (batch_size, N, channels, length, height)
# #         Sax  =  (Eax * input_tensor).permute(0, 4, 1, 2, 3).contiguous()          # Sax will now have a shape  (batch_size, N, channels, length, width)
# #         S = torch.cat((Scor, Ssag, Sax), dim = 1)                                 # Now S will have a shape of (batch_size, 3N, channels, length, length)
# #         S = S.view(-1,C,H,W).contiguous()
# #         pooled_feat = self.CNN_2D(S).view(B, 3*H, -1)            
# #         output_tensor = self.non_linear_proj(pooled_feat)          
# #         return output_tensor


# # class EmbeddingLayer(nn.Module):
# #     def __init__(self, emb_size: int = 256, total_tokens: int = 84):
# #         super(EmbeddingLayer, self).__init__()

# #         self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
# #         self.sep_token = nn.Parameter(torch.randn(1,1, emb_size))
# #         self.coronal_plane = nn.Parameter(torch.randn(1, emb_size))
# #         self.sagittal_plane = nn.Parameter(torch.randn(1, emb_size))
# #         self.axial_plane = nn.Parameter(torch.randn(1, emb_size))
# #         self.positions = nn.Parameter(torch.randn(total_tokens + 4, emb_size))

# #     def forward(self, input_tensor):
# #         b, _, _ = input_tensor.shape
# #         cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
# #         sep_token = repeat(self.sep_token, '() n e -> b n e', b=b)

# #         x = torch.cat((cls_tokens, input_tensor[:, :28, :], sep_token, input_tensor[:, 28:56, :], sep_token, input_tensor[:, 56:, :], sep_token), dim=1)

# #         x[:, :30] += self.coronal_plane
# #         x[:, 30:59] += self.sagittal_plane
# #         x[:, 59:] += self.axial_plane

# #         x += self.positions
       
# #         return x

# # class Slice_Extractor(nn.Module):
# #     def __init__(self, out_channels: int, emb_size: int = 256, total_tokens: int = 84):
# #         super(Slice_Extractor, self).__init__()

# #         # 2D CNN part
# #         # Load the pre-trained ResNet-50 model and replace the first convolutional layer and the fully connected layer
# #         self.CNN_2D = models.resnet50(weights=True)
# #         self.CNN_2D.conv1 = nn.Conv2d(out_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
# #         self.CNN_2D.fc = nn.Identity()

# #         # Non-Linear Projection block
# #         self.non_linear_proj = nn.Sequential(
# #             nn.Linear(2048, 512),
# #             nn.ReLU(),
# #             nn.Linear(512, emb_size)
# #         )

# #         # Embedding Layer parameters
# #         self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
# #         self.sep_token = nn.Parameter(torch.randn(1, 1, emb_size))
# #         self.coronal_plane = nn.Parameter(torch.randn(1, emb_size))
# #         self.sagittal_plane = nn.Parameter(torch.randn(1, emb_size))
# #         self.axial_plane = nn.Parameter(torch.randn(1, emb_size))
# #         self.positions = nn.Parameter(torch.randn(total_tokens + 4, emb_size))

# #     def forward(self, input_tensor):
# #         B, C, D, H, W = input_tensor.shape

# #         # Extract coronal features
# #         coronal_slices = torch.split(input_tensor, 1, dim=2)  # Split along depth (D)
# #         Ecor = torch.cat(coronal_slices, dim=2)

# #         # Extract sagittal features
# #         sagittal_slices = torch.split(input_tensor.clone(), 1, dim=3)  # Split along height (H)
# #         Esag = torch.cat(sagittal_slices, dim=3)

# #         # Extract axial features
# #         axial_slices = torch.split(input_tensor.clone(), 1, dim=4)  # Split along width (W)
# #         Eax = torch.cat(axial_slices, dim=4)

# #         # Calculate S using E for X
# #         Scor = (Ecor * input_tensor).permute(0, 2, 1, 3, 4).contiguous()  # Shape: (B, N, C, H, W)
# #         Ssag = (Esag * input_tensor).permute(0, 3, 1, 2, 4).contiguous()  # Shape: (B, N, C, D, W)
# #         Sax = (Eax * input_tensor).permute(0, 4, 1, 2, 3).contiguous()    # Shape: (B, N, C, D, H)

# #         # Concatenate the reshaped extracted features
# #         S = torch.cat((Scor, Ssag, Sax), dim=1)  # Shape: (B, 3N, C, H, W)

# #         # Flatten S for CNN input
# #         S = S.view(-1, C, H, W).contiguous()  # Shape: (B * 3N, C, H, W)

# #         # Pass through 2D CNN
# #         cnn_features = self.CNN_2D(S)  # Shape: (B * 3N, CNN_output_dim)

# #         # Reshape and apply non-linear projection
# #         cnn_features = cnn_features.view(B, 3 * H, -1)  # Shape: (B, 3N, CNN_output_dim)
# #         projected_features = self.non_linear_proj(cnn_features)  # Shape: (B, 3N, emb_size)

# #         # Embedding Layer logic
# #         b, n, e = projected_features.shape  # b: batch size, n: number of tokens, e: embedding size

# #         cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
# #         sep_token = repeat(self.sep_token, '() n e -> b n e', b=b)

# #         x = torch.cat((
# #             cls_tokens,
# #             projected_features[:, :H, :],  # Coronal tokens
# #             sep_token,
# #             projected_features[:, H:2*H, :],  # Sagittal tokens
# #             sep_token,
# #             projected_features[:, 2*H:, :],  # Axial tokens
# #             sep_token
# #         ), dim=1)  # Shape: (B, total_tokens + 4, emb_size)

# #         # Add plane embeddings
# #         x[:, :H+2] += self.coronal_plane  # First H tokens + cls_token and first sep_token
# #         x[:, H+2:2*H+3] += self.sagittal_plane  # Next H tokens + sep_token
# #         x[:, 2*H+3:] += self.axial_plane  # Last H tokens + sep_token

# #         # Add position embeddings
# #         x += self.positions

# #         return x  # Output shape: (B, total_tokens + 4, emb_size)


# class SliceExtractorWithEmbedding(nn.Module):
#     def __init__(self, input_channels: int, embedding_dim: int = 256, num_tokens: int = 84):
#         super(SliceExtractorWithEmbedding, self).__init__()

#         # Load pre-trained ResNet-50 and modify the input and output layers
#         self.resnet_2d = models.resnet50(pretrained=True)
#         self.resnet_2d.conv1 = nn.Conv2d(
#             in_channels=input_channels,
#             out_channels=64,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             bias=False
#         )
#         self.resnet_2d.fc = nn.Identity()

#         # Non-linear projection layers
#         self.projection = nn.Sequential(
#             nn.Linear(2048, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, embedding_dim)
#         )

#         # Embedding tokens and positional embeddings
#         self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
#         self.sep_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
#         self.plane_embeddings = nn.Parameter(torch.randn(3, embedding_dim))  # For coronal, sagittal, axial
#         self.positional_embeddings = nn.Parameter(torch.randn(num_tokens + 4, embedding_dim))

#     def forward(self, x):
#         batch_size, channels, depth, height, width = x.shape

#         # Extract slices along each plane
#         coronal_slices = [x[:, :, i, :, :] for i in range(depth)]         # List of [B, C, H, W]
#         sagittal_slices = [x[:, :, :, i, :] for i in range(height)]       # List of [B, C, D, W]
#         axial_slices = [x[:, :, :, :, i] for i in range(width)]           # List of [B, C, D, H]

#         # Stack slices and reshape for CNN input
#         coronal_stack = torch.stack(coronal_slices, dim=1)                # [B, D, C, H, W]
#         sagittal_stack = torch.stack(sagittal_slices, dim=1)              # [B, H, C, D, W]
#         axial_stack = torch.stack(axial_slices, dim=1)                    # [B, W, C, D, H]

#         # Reshape to (B * N, C, H, W) for CNN processing
#         coronal_input = coronal_stack.view(-1, channels, height, width)
#         sagittal_input = sagittal_stack.view(-1, channels, depth, width)
#         axial_input = axial_stack.view(-1, channels, depth, height)

#         # Pass through 2D CNN
#         coronal_features = self.resnet_2d(coronal_input)    # [B * D, 2048]
#         sagittal_features = self.resnet_2d(sagittal_input)  # [B * H, 2048]
#         axial_features = self.resnet_2d(axial_input)        # [B * W, 2048]

#         # Apply non-linear projection
#         coronal_proj = self.projection(coronal_features)    # [B * D, embedding_dim]
#         sagittal_proj = self.projection(sagittal_features)  # [B * H, embedding_dim]
#         axial_proj = self.projection(axial_features)        # [B * W, embedding_dim]

#         # Reshape back to (B, N, embedding_dim)
#         coronal_tokens = coronal_proj.view(batch_size, depth, -1)
#         sagittal_tokens = sagittal_proj.view(batch_size, height, -1)
#         axial_tokens = axial_proj.view(batch_size, width, -1)

#         # Concatenate tokens with cls and sep tokens
#         tokens = torch.cat([
#             self.cls_token.repeat(batch_size, 1, 1),         # [B, 1, embedding_dim]
#             coronal_tokens,                                  # [B, D, embedding_dim]
#             self.sep_token.repeat(batch_size, 1, 1),         # [B, 1, embedding_dim]
#             sagittal_tokens,                                 # [B, H, embedding_dim]
#             self.sep_token.repeat(batch_size, 1, 1),         # [B, 1, embedding_dim]
#             axial_tokens,                                    # [B, W, embedding_dim]
#             self.sep_token.repeat(batch_size, 1, 1)          # [B, 1, embedding_dim]
#         ], dim=1)  # Expected shape: [B, total_tokens + 4, embedding_dim] = [B, 88, embedding_dim]

#         # Apply plane embeddings
#         tokens_without_cls_sep = tokens[:, 1:, :]  # Exclude the cls_token

#         # Create plane_embeddings tensor
#         plane_embeddings = torch.zeros(batch_size, tokens_without_cls_sep.size(1), tokens_without_cls_sep.size(2), device=x.device)

#         # Define indices for each plane in tokens_without_cls_sep
#         coronal_indices = torch.arange(0, depth)  # Positions 0 to 27
#         sep1_index = depth                        # Position 28

#         sagittal_start = depth + 1                # Position 29
#         sagittal_indices = torch.arange(sagittal_start, sagittal_start + height)  # Positions 29 to 56
#         sep2_index = sagittal_start + height      # Position 57

#         axial_start = sep2_index + 1              # Position 58
#         axial_indices = torch.arange(axial_start, axial_start + width)            # Positions 58 to 85
#         sep3_index = axial_start + width          # Position 86

#         # Assign plane embeddings
#         plane_embeddings[:, coronal_indices, :] = self.plane_embeddings[0]
#         plane_embeddings[:, sagittal_indices, :] = self.plane_embeddings[1]
#         plane_embeddings[:, axial_indices, :] = self.plane_embeddings[2]
#         # Sep tokens and cls token remain without plane embeddings

#         # Add plane embeddings to tokens_without_cls_sep
#         tokens_without_cls_sep += plane_embeddings

#         # Reconstruct tokens with cls_token
#         tokens = torch.cat([tokens[:, :1, :], tokens_without_cls_sep], dim=1)

#         # Add positional embeddings
#         tokens += self.positional_embeddings

#         return tokens

# #old
# # class SliceExtractorWithEmbedding(nn.Module):
# #     def __init__(self, input_channels: int, embedding_dim: int = 256, num_tokens: int = 84):
# #         super(SliceExtractorWithEmbedding, self).__init__()

# #         # Load pre-trained ResNet-50 and modify the input and output layers
# #         self.resnet_2d = models.resnet50(pretrained=True)
# #         self.resnet_2d.conv1 = nn.Conv2d(
# #             in_channels=input_channels,
# #             out_channels=64,
# #             kernel_size=7,
# #             stride=2,
# #             padding=3,
# #             bias=False
# #         )
# #         self.resnet_2d.fc = nn.Identity()

# #         # Non-linear projection layers
# #         self.projection = nn.Sequential(
# #             nn.Linear(2048, 512),
# #             nn.ReLU(inplace=True),
# #             nn.Linear(512, embedding_dim)
# #         )

# #         # Embedding tokens and positional embeddings
# #         self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
# #         self.sep_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
# #         self.plane_embeddings = nn.Parameter(torch.randn(3, embedding_dim))  # For coronal, sagittal, axial
# #         self.positional_embeddings = nn.Parameter(torch.randn(num_tokens + 4, embedding_dim))

# #     def forward(self, x):
# #         batch_size, channels, depth, height, width = x.shape

# #         # Extract slices along each plane
# #         coronal_slices = [x[:, :, i, :, :] for i in range(depth)]         # List of [B, C, H, W]
# #         sagittal_slices = [x[:, :, :, i, :] for i in range(height)]       # List of [B, C, D, W]
# #         axial_slices = [x[:, :, :, :, i] for i in range(width)]           # List of [B, C, D, H]

# #         # Stack slices and reshape for CNN input
# #         coronal_stack = torch.stack(coronal_slices, dim=1)                # [B, D, C, H, W]
# #         sagittal_stack = torch.stack(sagittal_slices, dim=1)              # [B, H, C, D, W]
# #         axial_stack = torch.stack(axial_slices, dim=1)                    # [B, W, C, D, H]

# #         # Reshape to (B * N, C, H, W) for CNN processing
# #         coronal_input = coronal_stack.view(-1, channels, height, width)
# #         sagittal_input = sagittal_stack.view(-1, channels, depth, width)
# #         axial_input = axial_stack.view(-1, channels, depth, height)

# #         # Pass through 2D CNN
# #         coronal_features = self.resnet_2d(coronal_input)    # [B * D, 2048]
# #         sagittal_features = self.resnet_2d(sagittal_input)  # [B * H, 2048]
# #         axial_features = self.resnet_2d(axial_input)        # [B * W, 2048]

# #         # Apply non-linear projection
# #         coronal_proj = self.projection(coronal_features)    # [B * D, embedding_dim]
# #         sagittal_proj = self.projection(sagittal_features)  # [B * H, embedding_dim]
# #         axial_proj = self.projection(axial_features)        # [B * W, embedding_dim]

# #         # Reshape back to (B, N, embedding_dim)
# #         coronal_tokens = coronal_proj.view(batch_size, depth, -1)
# #         sagittal_tokens = sagittal_proj.view(batch_size, height, -1)
# #         axial_tokens = axial_proj.view(batch_size, width, -1)

# #         # Concatenate tokens with cls and sep tokens
# #         tokens = torch.cat([
# #             self.cls_token.repeat(batch_size, 1, 1),
# #             coronal_tokens,
# #             self.sep_token.repeat(batch_size, 1, 1),
# #             sagittal_tokens,
# #             self.sep_token.repeat(batch_size, 1, 1),
# #             axial_tokens,
# #             self.sep_token.repeat(batch_size, 1, 1)
# #         ], dim=1)  # [B, total_tokens + 4, embedding_dim]

# #         # Add plane embeddings
# #         plane_embeddings = torch.cat([
# #             self.plane_embeddings[0].unsqueeze(0).unsqueeze(0).repeat(batch_size, depth + 1, 1),  # coronal + cls
# #             self.plane_embeddings[1].unsqueeze(0).unsqueeze(0).repeat(batch_size, height + 1, 1),  # sagittal + sep
# #             self.plane_embeddings[2].unsqueeze(0).unsqueeze(0).repeat(batch_size, width + 1, 1)    # axial + sep
# #         ], dim=1)

# #         # Apply plane embeddings (skip cls and sep tokens)
# #         tokens_without_cls_sep = tokens[:, 1:, :]
# #         tokens_without_cls_sep += plane_embeddings[:, :-1, :]
# #         tokens = torch.cat([tokens[:, :1, :], tokens_without_cls_sep], dim=1)

# #         # Add positional embeddings
# #         tokens += self.positional_embeddings

# #         return tokens
    

# class MSA(nn.Module):
#     def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout_rate: float = 0.0):
#         super(MSA, self).__init__()
#         assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

#         self.embed_dim = embed_dim      # Total embedding dimension
#         self.num_heads = num_heads      # Number of attention heads
#         self.head_dim = embed_dim // num_heads  # Dimension per head

#         # Linear layers for query, key, and value projections
#         self.qkv_projection = nn.Linear(embed_dim, embed_dim * 3)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.output_projection = nn.Linear(embed_dim, embed_dim)

#     def forward(self, x, mask=None):
#         """
#         Forward pass for multi-head self-attention.

#         Args:
#             x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)
#             mask (Tensor, optional): Attention mask of shape (batch_size, seq_length). Default is None.

#         Returns:
#             Tensor: Output tensor of shape (batch_size, seq_length, embed_dim)
#         """
#         batch_size, seq_length, embed_dim = x.size()

#         # Project input embeddings to queries, keys, and values
#         qkv = self.qkv_projection(x)  # Shape: (batch_size, seq_length, embed_dim * 3)

#         # Reshape and permute to separate heads
#         qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
#         qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: (3, batch_size, num_heads, seq_length, head_dim)

#         # Unpack queries, keys, and values
#         queries, keys, values = qkv[0], qkv[1], qkv[2]  # Each of shape: (batch_size, num_heads, seq_length, head_dim)

#         # Compute scaled dot-product attention scores
#         scaling_factor = self.head_dim ** 0.5
#         attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / scaling_factor  # Shape: (batch_size, num_heads, seq_length, seq_length)

#         # Apply mask if provided
#         if mask is not None:
#             # mask shape: (batch_size, seq_length)
#             # Expand mask to match attention_scores shape
#             mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_length)
#             attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

#         # Compute attention weights
#         attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: (batch_size, num_heads, seq_length, seq_length)
#         attention_weights = self.dropout(attention_weights)

#         # Compute context vector as weighted sum of values
#         context = torch.matmul(attention_weights, values)  # Shape: (batch_size, num_heads, seq_length, head_dim)

#         # Concatenate heads and reshape
#         context = context.transpose(1, 2).contiguous()  # Shape: (batch_size, seq_length, num_heads, head_dim)
#         context = context.view(batch_size, seq_length, embed_dim)  # Shape: (batch_size, seq_length, embed_dim)

#         # Final linear projection
#         output = self.output_projection(context)  # Shape: (batch_size, seq_length, embed_dim)

#         return output
    

# class VanillaBlock(nn.Module):
#     def __init__(self,
#                  embed_dim: int = 256,
#                  num_heads: int = 8,
#                  dropout_rate: float = 0.0,
#                  forward_expansion: int = 2,
#                  forward_dropout_rate: float = 0.0,
#                  **kwargs):
#         super(VanillaBlock, self).__init__()

#         # Layer normalization and multi-head self-attention
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.attention = MSA(embed_dim=embed_dim, num_heads=num_heads, dropout_rate=dropout_rate, **kwargs)
#         self.dropout1 = nn.Dropout(dropout_rate)

#         # Feedforward network (FFN)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.feedforward = nn.Sequential(
#             nn.Linear(embed_dim, forward_expansion * embed_dim),
#             nn.GELU(),
#             nn.Dropout(forward_dropout_rate),
#             nn.Linear(forward_expansion * embed_dim, embed_dim),
#             nn.Dropout(dropout_rate),
#         )

#     def forward(self, x):
#         """
#         Forward pass for the Transformer encoder block.

#         Args:
#             x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)

#         Returns:
#             Tensor: Output tensor of shape (batch_size, seq_length, embed_dim)
#         """
#         # First residual connection with multi-head attention
#         x_residual = x
#         x = self.norm1(x)
#         x = self.attention(x)
#         x = x_residual + self.dropout1(x)

#         # Second residual connection with feedforward network
#         x_residual = x
#         x = self.norm2(x)
#         x = self.feedforward(x)
#         x = x_residual + x  # Residual connection (dropout included in feedforward)

#         return x




# # class FeedForwardBlock(nn.Sequential):
# #     def __init__(self, emb_size: int, expansion: int = 2, drop_p: float = 0.):
# #         super().__init__(
# #             nn.Linear(emb_size, expansion * emb_size),
# #             nn.GELU(),
# #             nn.Dropout(drop_p),
# #             nn.Linear(expansion * emb_size, emb_size),
# #         )



# # class TransformerEncoderBlock(nn.Sequential):

# #     def __init__(self,
# #                  emb_size: int = 256,
# #                  drop_p: float = 0.,
# #                  forward_expansion: int = 2,
# #                  forward_drop_p: float = 0.,
# #                  ** kwargs):
# #         super().__init__()

# #         # Layer normalization and multi-head self-attention
# #         self.norm1 = nn.LayerNorm(emb_size)
# #         self.attention = MSA(emb_size, **kwargs)
# #         self.dropout1 = nn.Dropout(drop_p)

# #         # Feedforward block
# #         self.norm2 = nn.LayerNorm(emb_size)
# #         self.feedforward = FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
# #         self.dropout2 = nn.Dropout(drop_p)

# #     def forward(self, x):
# #         # Apply normalization, multi-head attention, and residual connection
# #         norm_x = self.norm1(x)
# #         x = self.attention(norm_x)
# #         x = x + self.dropout1(x)  # Residual connection
        
# #         # Apply normalization, feedforward block, and residual connection
# #         norm_x = self.norm2(x)
# #         feedforward_output = self.feedforward(norm_x)
# #         x = x + self.dropout2(feedforward_output)  # Residual connection

# #         return x
            

# class VanillaEncoder(nn.Module):
#     def __init__(self, depth: int = 8, **kwargs):
#         super().__init__()
#         self.layers = nn.ModuleList([VanillaBlock(**kwargs) for _ in range(depth)])

#     def forward(self, x):
#         #attention_maps = []
#         for layer in self.layers:
#             x = layer(x)
#         return x
    


# class ClassificationHead(nn.Module):

#     def __init__(self, emb_size: int = 256, n_classes: int = 2):
#         super().__init__()
#         self.linear = nn.Linear(emb_size, n_classes)

#     def forward(self, x):
#         cls_token = x[:, 0]
#         return self.linear(cls_token)#, attn_map
        
# class ThreeDViTMedNet(nn.Sequential):
#     def __init__(self,
#                 in_channels: int = 7,
#                 out_channels: int = 32,
#                 emb_size: int = 256,
#                 depth: int = 8,
#                 n_classes: int = 2,
#                 **kwargs):
#         super().__init__()

#         self.cnn_3d_layer = CNN_3D_Layer(in_channels, out_channels)
#         self.slice_extractor_with_embedding = SliceExtractorWithEmbedding(out_channels)
#         self.vanilla_encoder = VanillaEncoder(depth, emb_size=emb_size, **kwargs)
#         self.classification_head = ClassificationHead(emb_size, n_classes)
#     def forward(self, x):
#         x = shifting_image_procedure(x)
#         x = self.cnn_3d_layer(x)
#         x = self.slice_extractor_with_embedding(x)
#         x = self.vanilla_encoder(x)
#         x = self.classification_head(x)
#         return x
