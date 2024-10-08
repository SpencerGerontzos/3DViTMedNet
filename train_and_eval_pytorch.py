import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy
import torchsummary
from torchsummary import summary
from medmnist import FractureMNIST3D
import matplotlib.pyplot as plt
from einops import rearrange
import matplotlib.patches as patches

from models import resnet_models, ThreeD_SWIN, ThreeD_ViT, ThreeDViTMedNet

from models.ThreeDViTMedNet import ThreeDViTMedNet
from models.ThreeD_SWIN import SwinTransformer3D
from models.ThreeD_ViT import ViT
from models.resnet_models import ResNet18, ResNet50


import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter
from medmnist import INFO, Evaluator
# from models import ResNet18, ResNet50
# from models import SwinTransformer3D
# from vit import ViT
from scipy.ndimage import zoom
from skimage.transform import resize
from tensorboardX import SummaryWriter
from tqdm import trange
from utils import Transform3D, model_to_syncbn

# def plot_attention_map(attention_map, layer_idx, head_idx, token_idx):
#     """
#     Plots the attention map for a specific layer, head, and token.
#     Args:
#         attention_map (Tensor): The attention map tensor of shape (batch_size, num_heads, num_tokens, num_tokens)
#         layer_idx (int): The index of the transformer layer.
#         head_idx (int): The index of the attention head to visualize.
#         token_idx (int): The token index for which the attention map should be visualized.
#     """
#     attention = attention_map[layer_idx][0, head_idx, token_idx].detach().cpu().numpy()
#     plt.imshow(attention, cmap='viridis')
#     plt.colorbar()
#     plt.title(f'Attention map (Layer {layer_idx}, Head {head_idx}, Token {token_idx})')
#     plt.show()

import torch
import numpy as np
from torchvision.transforms import Compose

class RandomFlip3D:
    def __init__(self, axes=(0, 1, 2), p=0.5):
        self.axes = axes
        self.p = p

    def __call__(self, sample):
        image, label = sample
        for axis in self.axes:
            if np.random.rand() < self.p:
                image = np.flip(image, axis=axis).copy()
                # If labels need flipping, apply the same transformation
                # label = np.flip(label, axis=axis).copy()
        return image, label

class RandomRotate3D:
    def __init__(self, angles=(15, 15, 15)):
        self.angles = angles  # Max rotation angles in degrees

    def __call__(self, sample):
        image, label = sample
        angle_x = np.random.uniform(-self.angles[0], self.angles[0])
        angle_y = np.random.uniform(-self.angles[1], self.angles[1])
        angle_z = np.random.uniform(-self.angles[2], self.angles[2])
        # Apply rotations around each axis
        # You can use scipy.ndimage.rotate or implement custom rotation
        return image, label

class AddGaussianNoise3D:
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample
        noise = np.random.normal(self.mean, self.std, image.shape)
        image = image + noise
        return image, label
from torchvision.transforms import ToTensor

train_transforms = Compose([
    RandomFlip3D(axes=(0, 1, 2), p=0.5),
    RandomRotate3D(angles=(15, 15, 15)),
    AddGaussianNoise3D(mean=0.0, std=0.01),
    ToTensor(),
])




def plot_attention_map(attention_map, title):
    """
    Plots the attention map for a specific layer, head, and token.
    Args:
        attention_map (Tensor): The attention map tensor of shape (num_tokens, num_tokens)
        title (str): The title of the plot.
    """
    plt.imshow(attention_map, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()

def get_plane_attention_maps(attention_maps, num_slices_per_plane):
    """
    Extract attention maps for each plane (axial, sagittal, coronal).
    Args:
        attention_maps (Tensor): The attention map tensor of shape (batch_size, num_heads, num_tokens, num_tokens)
        num_slices_per_plane (int): The number of slices per plane (for example, 28 for 28x28x28 data)
    """
    axial_attention = attention_maps[:, :, :num_slices_per_plane, :num_slices_per_plane]  # Axial plane tokens
    sagittal_attention = attention_maps[:, :, num_slices_per_plane:2*num_slices_per_plane, num_slices_per_plane:2*num_slices_per_plane]  # Sagittal plane tokens
    coronal_attention = attention_maps[:, :, 2*num_slices_per_plane:, 2*num_slices_per_plane:]  # Coronal plane tokens

    return axial_attention, sagittal_attention, coronal_attention

def visualize_attention_for_planes(attention_maps, layer_idx, head_idx, num_slices_per_plane):
    """
    Visualize the attention maps for the axial, sagittal, and coronal planes.
    Args:
        attention_maps (Tensor): The attention map tensor of shape (batch_size, num_heads, num_tokens, num_tokens)
        layer_idx (int): The index of the transformer layer.
        head_idx (int): The index of the attention head.
        num_slices_per_plane (int): The number of slices per plane.
    """
    # Extract attention maps for the planes
    axial_attention, sagittal_attention, coronal_attention = get_plane_attention_maps(
        attention_maps[layer_idx][0], num_slices_per_plane)

    # Visualize attention maps for each plane
    plot_attention_map(axial_attention[head_idx].detach().cpu().numpy(), "Axial Plane Attention")
    plot_attention_map(sagittal_attention[head_idx].detach().cpu().numpy(), "Sagittal Plane Attention")
    plot_attention_map(coronal_attention[head_idx].detach().cpu().numpy(), "Coronal Plane Attention")


def visualize_attention(attn_submatrix, plane_name, img_slice, axs_row):
    # axs_row is a tuple of axes (ax_attn, ax_img)
    
    # Plot the attention matrix
    im0 = axs_row[0].imshow(attn_submatrix, cmap='viridis', aspect='equal')
    axs_row[0].set_title(f'{plane_name} Plane Attention')
    axs_row[0].set_xlabel('Key Positions')
    axs_row[0].set_ylabel('Query Positions')
    plt.colorbar(im0, ax=axs_row[0])

    # Plot the original image slice
    axs_row[1].imshow(img_slice, cmap='gray', aspect='equal')
    axs_row[1].set_title(f'{plane_name} Plane Original Slice')
    axs_row[1].axis('off')

def extract_and_visualize_intra_plane_attention(attn_matrix, indices, plane_name, img_slice, axs_row):
    attn_submatrix = attn_matrix[np.ix_(indices, indices)]
    # Normalize the attention weights
    # Sum over keys (columns) for each query (row)
    row_sums = attn_submatrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    # Normalize the submatrix
    attn_submatrix_normalized = attn_submatrix / row_sums
    visualize_attention(attn_submatrix_normalized, plane_name, img_slice, axs_row)



def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, gradcam, run):
    
    import torchio as tio
    train_transform = tio.Compose([
        # Random flips along acceptable axes
        tio.RandomFlip(
            axes=('LR',),         # Only if left-right flips are acceptable
            flip_probability=0.5,
        ),

        # Random rotations with small angles
        tio.RandomAffine(
            scales=1.0,
            degrees=(-10, 10),    # Rotate between -10 and +10 degrees
            translation=0,
            isotropic=True,
        ),

        # Random scaling
        tio.RandomAffine(
            scales=(0.95, 1.05),  # Scale between 95% and 105%
            degrees=0,
            translation=0,
            isotropic=True,
        ),

        # Random translation
        tio.RandomAffine(
            scales=1.0,
            degrees=0,
            translation=(-5, 5),  # Translate up to ±5 mm
            isotropic=True,
        ),

        # Adding Gaussian noise
        tio.RandomNoise(
            mean=0.0,
            std=(0, 0.025),
        ),

        # Random gamma correction
        tio.RandomGamma(
            log_gamma=(-0.1, 0.1),
        ),

        # Random bias field
        tio.RandomBiasField(
            coefficients=0.5,
        ),

        # Elastic deformation
        tio.RandomElasticDeformation(
            num_control_points=7,
            max_displacement=2.0,
            locked_borders=2,
        ),

        # Z-Normalization
        tio.ZNormalization(),
    ])
    
    is_resnet = False
    lr = 0.001
    gamma=0.1
    milestones = [.25 * num_epochs ,0.5 * num_epochs, 0.75 * num_epochs]

    info = INFO[data_flag]
    task = info['task']
    n_channels = 3 if as_rgb else info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])
    
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu') 
        
    output_root = os.path.join(output_root, data_flag, model_flag + str('-')+time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    print('==> Preparing data...')

    train_transform1 = Transform3D(mul='random') if shape_transform else Transform3D()
    eval_transform = Transform3D(mul='0.5') if shape_transform else Transform3D()
     
    train_dataset = DataClass(split='train', transform=train_transform1, download=download, as_rgb=as_rgb, size=size)
    train_dataset_at_eval = DataClass(split='train', transform=eval_transform, download=download, as_rgb=as_rgb, size=size)
    val_dataset = DataClass(split='val', transform=eval_transform, download=download, as_rgb=as_rgb, size=size)
    test_dataset = DataClass(split='test', transform=eval_transform, download=download, as_rgb=as_rgb, size=size)

    
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset_at_eval,
                                batch_size=batch_size,
                                shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)

    print('==> Building and training model...')

    if model_flag == 'resnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes)
        is_resnet = True
    elif model_flag == 'vit':
        model = ViT(
            image_size = size,          # image size
            frames = size,               # number of frames
            image_patch_size = 2,     # image patch size, Original ViT had patch size of 16x16 for a 224 image (14 times less)
            frame_patch_size = 2,      # frame patch size
            num_classes = n_classes,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        #model = model.to(device)
        #print(summary(model,(1, 28, 28, 28)))
    elif model_flag == 'medicalnet_transfer':
        model = medicalnet_transfer(
            n_classes= n_classes,
            pretrained_path = '/home/s224534582/MedicalNet/pretrain/resnet_50_23dataset.pth')
        
        # net_dict = model.state_dict()
        # print('prior to loading weights')
        # pretrain = torch.load('/home/s224534582/MedicalNet/pretrain/resnet_50_23dataset.pth')
        # # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        # pretrain_dict = {k.replace("module.", ""): v for k, v in pretrain['state_dict'].items() if k.replace("module.", "") in net_dict.keys()}
        # net_dict.update(pretrain_dict)
        # model.load_state_dict(net_dict)

        # print('pretrained weights loaded')
        # for param_tensor in model.state_dict():
        #     print(f"Layer: {param_tensor}, Loaded: {torch.equal(model.state_dict()[param_tensor], pretrain[param_tensor])}")

        new_parameters = [] 
        # for pname, p in model.named_parameters():
        #     for layer_name in opt.new_layer_names:
        #         if pname.find(layer_name) >= 0:
        #             new_parameters.append(p)
        #             break

        #new_parameters_id = list(map(id, new_parameters))
        #base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        #parameters = {'base_parameters': base_parameters, 
        #              'new_parameters': new_parameters}

    elif model_flag == '3DViTMedNet':
        model = ThreeDViTMedNet(
            n_classes= n_classes
        )


        # Load the pretrained weights into the model
        #state_dict = torch.load(model_path)

        #model.load_state_dict(state_dict, strict=False)  # strict=False allows loading if some layers do not match

    elif model_flag =='swin':
        # Create a 3D Swin Transformer model instance
        #model = SwinTransformer3D(num_classes=n_classes)
        #import torchvision.models.video.swin3d_t as swin3dt

        #model = swin3dt(num_classes = n_classes, )
        model = SwinTransformer3D(
            img_size=(28, 28, 28),
            patch_size=2,
            in_channels=1,
            num_classes=n_classes,  # Adjust based on your classification task
            embed_dim=96,
            depths=[2, 2],
            num_heads=[3, 6]
        )




        #model = model.to(device)
        #print(summary(model, (1, 28, 28, 28)))
    else:
        raise NotImplementedError
    

    if conv=='ACSConv':
        model = model_to_syncbn(ACSConverter(model))
    if conv=='Conv2_5d':
        model = model_to_syncbn(Conv2_5dConverter(model))
    if conv=='Conv3d':
        if pretrained_3d == 'i3d':
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=-3))
        else:
            model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=None))
    
    print('we have this many gpus avalible: ' + str(torch.cuda.device_count()))
    if torch.cuda.device_count() > 1: #checking if we have more than 1 gpu
	    model = nn.DataParalell(model)
    else:
	    model = model.to(device)


    train_evaluator = medmnist.Evaluator(data_flag, 'train', size=size)
    val_evaluator = medmnist.Evaluator(data_flag, 'val', size=size)
    test_evaluator = medmnist.Evaluator(data_flag, 'test', size=size)

    criterion = nn.CrossEntropyLoss()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=False)
        train_metrics = test(model, train_evaluator, train_loader_at_eval, criterion, device, run, output_root)
        val_metrics = test(model, val_evaluator, val_loader, criterion, device, run, output_root)
        test_metrics = test(model, test_evaluator, test_loader, criterion, device, run, output_root)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))
        
    if num_epochs == -2:#prints the grad-cam of the resnet-50 models
        print
        model.eval()#assuming resnet-50

        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Identify the target layer
        target_layer = model.layer4[-1].conv3  # Last conv layer in layer4

        # Initialize dictionaries to store activations and gradients
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations['value'] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0].detach()

        # Register the hooks
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        # Load your 28x28x28 image
        # Replace with your actual image data
        image_np = np.random.rand(28, 28, 28).astype(np.float32)

        # Prepare the input tensor
        input_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(device)

        # Forward pass
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        # Backward pass
        model.zero_grad()
        output[0, pred_class].backward()

        # Compute Grad-CAM
        activation = activations['value']
        gradient = gradients['value']
        weights = torch.mean(gradient, dim=(2, 3, 4), keepdim=True)
        grad_cam = torch.sum(weights * activation, dim=1)
        grad_cam = torch.relu(grad_cam)
        grad_cam -= grad_cam.min()
        grad_cam /= grad_cam.max()

        # Upsample Grad-CAM
        grad_cam = F.interpolate(
            grad_cam.unsqueeze(1), size=input_tensor.shape[2:], mode='trilinear', align_corners=False
        ).squeeze(1)

        # Convert tensors to NumPy arrays
        grad_cam_np = grad_cam.squeeze().cpu().numpy()
        input_np = input_tensor.squeeze().cpu().numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())

        # Select slices
        slice_idx = 14  # Middle slice

        axial_image = input_np[:, :, slice_idx]
        axial_cam = grad_cam_np[:, :, slice_idx]

        coronal_image = input_np[:, slice_idx, :]
        coronal_cam = grad_cam_np[:, slice_idx, :]

        sagittal_image = input_np[slice_idx, :, :]
        sagittal_cam = grad_cam_np[slice_idx, :, :]

        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(axial_image, cmap='gray')
        axes[0].imshow(axial_cam, cmap='jet', alpha=0.5)
        axes[0].set_title('Axial Plane')
        axes[0].axis('off')

        axes[1].imshow(coronal_image, cmap='gray')
        axes[1].imshow(coronal_cam, cmap='jet', alpha=0.5)
        axes[1].set_title('Coronal Plane')
        axes[1].axis('off')

        axes[2].imshow(sagittal_image, cmap='gray')
        axes[2].imshow(sagittal_cam, cmap='jet', alpha=0.5)
        axes[2].set_title('Sagittal Plane')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(f'{data_flag.lower()}_resnet_attnmap_visualization.png')
        plt.close()
        return



        
    elif num_epochs == -1: #prints original image in c,a,s slices
        inputs, targets = next(iter(train_loader))

        # Move input to the device (e.g., GPU)
        inputs = inputs.to(device)

        input_image = inputs[0].detach().cpu().numpy()  # Shape: [channels, D, H, W]

        # Use the first channel if multiple channels
        input_image = input_image[0]  # Shape: [D, H, W]

        # Get the dimensions
        D, H, W = input_image.shape  # Should be (28, 28, 28)

        # Extract middle slices for each plane
        coronal_slice = input_image[:, H // 2, :]     # Shape: [D, W] => [28, 28]
        sagittal_slice = input_image[:, :, W // 2]    # Shape: [D, H] => [28, 28]
        axial_slice = input_image[D // 2, :, :]       # Shape: [H, W] => [28, 28]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot coronal plane slice
        axes[0].imshow(coronal_slice, cmap='gray')
        axes[0].set_title('Coronal Slice')
        axes[0].axis('off')

        # Plot sagittal plane slice
        axes[1].imshow(sagittal_slice, cmap='gray')
        axes[1].set_title('Sagittal Slice')
        axes[1].axis('off')

        # Plot axial plane slice
        axes[2].imshow(axial_slice, cmap='gray')
        axes[2].set_title('Axial Slice')
        axes[2].axis('off')

        # Show the figure
        plt.tight_layout()
        plt.savefig(f'{data_flag.lower()}_our_model_visualization.png')
        plt.close()
        return

    if num_epochs == 0: #prints attention map alongside original image
        # Get attention map
        model.eval()  # Set the model to evaluation mode

        # Get a single batch from the DataLoader
        inputs, targets = next(iter(train_loader))

        # Move input to the device (e.g., GPU)
        inputs = inputs.to(device)

        # Pass input through the 3D CNN
        threed_convs = model.cnnblock3d(inputs)

        # Pass convs through to mpms extractor
        mpms = model.mpms(threed_convs)

        # Get embeddings
        emb = model.embedding_layer(mpms)

        # Compute qkv
        qkv = model.transformer_encoder.layers[0].attention.qkv(emb)

        print("qkv shape:", qkv.shape)

        num_heads = model.transformer_encoder.layers[0].attention.num_heads
        emb_size = model.transformer_encoder.layers[0].attention.emb_size
        head_dim = emb_size // num_heads  # Dimension per head

        # Rearrange qkv to separate queries, keys, values
        qkv = rearrange(
            qkv, 'b n (qkv h d) -> qkv b h n d', h=num_heads, qkv=3, d=head_dim
        )
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # Step 3: Compute attention scores (energy)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # Shape: [batch_size, num_heads, seq_len, seq_len]

        print("Energy shape:", energy.shape)

        # Step 4: Apply scaling and softmax
        scaling = emb_size ** 0.5
        energy = energy / scaling
        attention_weights = F.softmax(energy, dim=-1)  # Shape: [batch_size, num_heads, seq_len, seq_len]

        # Select the first sample
        attn_weights_first_sample = attention_weights[0]  # Shape: [num_heads, seq_len, seq_len]

        # Average over heads
        attn_matrix = attn_weights_first_sample.mean(dim=0).detach().cpu().numpy()  # Shape: (88, 88)

        # Define token indices for each plane
        coronal_indices = list(range(1, 29))    # Positions 1-28
        sagittal_indices = list(range(30, 58))  # Positions 30-57
        axial_indices = list(range(59, 87))     # Positions 59-86

        # Extract original image slices
        input_image = inputs[0].detach().cpu().numpy()  # Shape: [channels, D, H, W]

        # Use the first channel if multiple channels
        input_image = input_image[0]  # Shape: [D, H, W]

        # Get the dimensions
        D, H, W = input_image.shape  # Should be (28, 28, 28)

        # Extract middle slices for each plane
        coronal_slice = input_image[:, H // 2, :]     # Shape: [D, W] => [28, 28]
        sagittal_slice = input_image[:, :, W // 2]    # Shape: [D, H] => [28, 28]
        axial_slice = input_image[D // 2, :, :]       # Shape: [H, W] => [28, 28]

        fig, axs = plt.subplots(3, 2, figsize=(12, 18))

        # Adjust spacing
        plt.subplots_adjust(hspace=0.4)

        # Coronal Plane
        extract_and_visualize_intra_plane_attention(
            attn_matrix, coronal_indices, 'Coronal', coronal_slice, axs_row=axs[0]
        )

        # Sagittal Plane
        extract_and_visualize_intra_plane_attention(
            attn_matrix, sagittal_indices, 'Sagittal', sagittal_slice, axs_row=axs[1]
        )

        # Axial Plane
        extract_and_visualize_intra_plane_attention(
            attn_matrix, axial_indices, 'Axial', axial_slice, axs_row=axs[2]
        )

        # Save the combined figure
        plt.tight_layout()
        plt.savefig(f'{data_flag.lower()}attention_and_image_slices.png')
        plt.close()

        # Function to visualize attention matrices and original image slices
        # def visualize_attention(attn_submatrix, plane_name, img_slice):
        #     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            
        #     # Plot the attention matrix
        #     im0 = axs[0].imshow(attn_submatrix, cmap='viridis', aspect='equal')
        #     axs[0].set_title(f'{plane_name} Plane Attention')
        #     axs[0].set_xlabel('Key Positions')
        #     axs[0].set_ylabel('Query Positions')
        #     plt.colorbar(im0, ax=axs[0])
            
        #     # Plot the original image slice
        #     axs[1].imshow(img_slice, cmap='gray', aspect='equal')
        #     axs[1].set_title(f'{plane_name} Plane Original Slice')
        #     axs[1].axis('off')
            
        #     plt.tight_layout()
        #     plt.savefig(f'{plane_name.lower()}_attention_and_image_l.png')
        #     plt.close()

        # # Extract and visualize intra-plane attention
        # def extract_and_visualize_intra_plane_attention(attn_matrix, indices, plane_name, img_slice):
        #     attn_submatrix = attn_matrix[np.ix_(indices, indices)]
        #     # Normalize the attention weights
        #     # Sum over keys (columns) for each query (row)
        #     row_sums = attn_submatrix.sum(axis=1, keepdims=True)
            
        #     # Avoid division by zero
        #     row_sums[row_sums == 0] = 1.0
            
        #     # Normalize the submatrix
        #     attn_submatrix_normalized = attn_submatrix / row_sums
        #     visualize_attention(attn_submatrix_normalized, plane_name, img_slice)

        # Coronal Plane
        #extract_and_visualize_intra_plane_attention(attn_matrix, coronal_indices, 'Coronal', coronal_slice)

        # Sagittal Plane
        #extract_and_visualize_intra_plane_attention(attn_matrix, sagittal_indices, 'Sagittal', sagittal_slice)

        # Axial Plane
        #extract_and_visualize_intra_plane_attention(attn_matrix, axial_indices, 'Axial', axial_slice)
        return

    # else:
    #     # Get attention map
    #     model.eval()  # Set the model to evaluation mode

    #     # Get a single batch from the DataLoader
    #     inputs, targets = next(iter(train_loader))

    #     # Move input to the device (e.g., GPU)
    #     inputs = inputs.to(device)

    #     # Pass input through the 3D CNN
    #     threed_convs = model.cnnblock3d(inputs)

    #     # Pass convs through to mpms extractor
    #     mpms = model.mpms(threed_convs)

    #     # Get embeddings
    #     emb = model.embedding_layer(mpms)

    #     # Compute qkv
    #     qkv = model.transformer_encoder.layers[0].attention.qkv(emb)

    #     print("qkv shape:", qkv.shape)

    #     num_heads = model.transformer_encoder.layers[0].attention.num_heads
    #     emb_size = model.transformer_encoder.layers[0].attention.emb_size
    #     head_dim = emb_size // num_heads  # Dimension per head

    #     # Rearrange qkv to separate queries, keys, values
    #     qkv = rearrange(
    #         qkv, 'b n (qkv h d) -> qkv b h n d', h=num_heads, qkv=3, d=head_dim
    #     )
    #     queries, keys, values = qkv[0], qkv[1], qkv[2]

    #     # Step 3: Compute attention scores (energy)
    #     energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # Shape: [batch_size, num_heads, seq_len, seq_len]

    #     print("Energy shape:", energy.shape)

    #     # Step 4: Apply scaling and softmax
    #     scaling = emb_size ** 0.5
    #     energy = energy / scaling
    #     attention_weights = F.softmax(energy, dim=-1)  # Shape: [batch_size, num_heads, seq_len, seq_len]

    #     # Select the first sample
    #     attn_weights_first_sample = attention_weights[0]  # Shape: [num_heads, seq_len, seq_len]

    #     # Average over heads
    #     attn_matrix = attn_weights_first_sample.mean(dim=0).detach().cpu().numpy()  # Shape: (88, 88)

    #     # Define token indices for each plane
    #     coronal_indices = list(range(1, 29))   # Positions 1-28
    #     sagittal_indices = list(range(30, 58)) # Positions 30-57
    #     axial_indices = list(range(59, 87))    # Positions 59-86

    #     # Function to normalize attention submatrix
    #     def normalize_attention_submatrix(attn_submatrix):
    #         # Sum over keys (columns) for each query (row)
    #         row_sums = attn_submatrix.sum(axis=1, keepdims=True)
    #         # Avoid division by zero
    #         row_sums[row_sums == 0] = 1.0
    #         # Normalize the submatrix
    #         attn_submatrix_normalized = attn_submatrix / row_sums
    #         return attn_submatrix_normalized

    #     # Function to get attention scores
    #     def get_attention_scores(attn_submatrix_normalized):
    #         # Sum over keys to get attention scores for each query
    #         attention_scores = attn_submatrix_normalized.sum(axis=1)  # Shape: [num_tokens]
    #         return attention_scores

    #     # Function to normalize scores
    #     def normalize_scores(scores):
    #         # Use np.ptp() instead of scores.ptp()
    #         return (scores - scores.min()) / np.ptp(scores)

    #     # Function to map attention scores to image slices
    #     def map_attention_to_slice(attention_scores_norm, img_slice, num_tokens):
    #         # Resize attention scores to match the dimension of the slice
    #         zoom_factor = img_slice.shape[1] / num_tokens  # Adjust axis based on orientation
    #         attention_map = zoom(attention_scores_norm, zoom_factor)
    #         # Tile or reshape to match the image slice dimensions
    #         attention_map = np.tile(attention_map, (img_slice.shape[0], 1))
    #         return attention_map

    #     # Extract and process attention for each plane
    #     def extract_attention_for_plane(attn_matrix, indices, plane_name):
    #         attn_submatrix = attn_matrix[np.ix_(indices, indices)]
    #         # Normalize the attention weights
    #         attn_submatrix_normalized = normalize_attention_submatrix(attn_submatrix)
    #         # Get attention scores
    #         attention_scores = get_attention_scores(attn_submatrix_normalized)
    #         # Normalize attention scores for visualization
    #         attention_scores_norm = normalize_scores(attention_scores)
    #         return attention_scores_norm

    #     # Get attention scores for each plane
    #     coronal_attention_scores_norm = extract_attention_for_plane(attn_matrix, coronal_indices, 'Coronal')
    #     sagittal_attention_scores_norm = extract_attention_for_plane(attn_matrix, sagittal_indices, 'Sagittal')
    #     axial_attention_scores_norm = extract_attention_for_plane(attn_matrix, axial_indices, 'Axial')

    #     # Extract original image slices
    #     input_image = inputs[0].detach().cpu().numpy()  # Shape: [channels, D, H, W]

    #     # Use the first channel if multiple channels
    #     input_image = input_image[0]  # Shape: [D, H, W]

    #     # Get the dimensions
    #     D, H, W = input_image.shape

    #     # Extract middle slices for each plane
    #     coronal_slice = input_image[:, H // 2, :].T  # Shape: [W, D]
    #     sagittal_slice = input_image[:, :, W // 2].T  # Shape: [H, D]
    #     axial_slice = input_image[D // 2, :, :]  # Shape: [H, W]

    #     # Map attention scores to image slices
    #     coronal_attention_map = map_attention_to_slice(coronal_attention_scores_norm, coronal_slice, len(coronal_attention_scores_norm))
    #     sagittal_attention_map = map_attention_to_slice(sagittal_attention_scores_norm, sagittal_slice, len(sagittal_attention_scores_norm))
    #     axial_attention_map = map_attention_to_slice(axial_attention_scores_norm, axial_slice, len(axial_attention_scores_norm))

    #     # Function to visualize attention maps next to original images
    #     def visualize_attention_with_image(img_slice, attn_map, plane_name):
    #         fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    #         # Original Image Slice
    #         axs[0].imshow(img_slice, cmap='gray', aspect='auto')
    #         axs[0].set_title(f'{plane_name} Plane Original Slice')
    #         axs[0].axis('off')
            
    #         # Attention Map Overlay
    #         axs[1].imshow(img_slice, cmap='gray', aspect='auto')
    #         axs[1].imshow(attn_map, cmap='jet', alpha=0.5, aspect='auto')
    #         axs[1].set_title(f'{plane_name} Plane Attention Map')
    #         axs[1].axis('off')

    #         plt.tight_layout()
    #         plt.savefig(f'{plane_name.lower()}_plane_attention_with_image.png')
    #         plt.close()

    #     # Visualize attention maps next to original images for each plane
    #     visualize_attention_with_image(coronal_slice, coronal_attention_map, 'Coronal')
    #     visualize_attention_with_image(sagittal_slice, sagittal_attention_map, 'Sagittal')
    #     visualize_attention_with_image(axial_slice, axial_attention_map, 'Axial')

    #     # Alternatively, create a composite figure with all planes
    #     fig, axs = plt.subplots(3, 2, figsize=(12, 18))

    #     planes = ['Coronal', 'Sagittal', 'Axial']
    #     slices = [coronal_slice, sagittal_slice, axial_slice]
    #     attention_maps = [coronal_attention_map, sagittal_attention_map, axial_attention_map]

    #     for i, (plane, img_slice, attn_map) in enumerate(zip(planes, slices, attention_maps)):
    #         # Original Image Slice
    #         axs[i, 0].imshow(img_slice, cmap='gray', aspect='auto')
    #         axs[i, 0].set_title(f'{plane} Plane Original Slice')
    #         axs[i, 0].axis('off')
            
    #         # Attention Map Overlay
    #         axs[i, 1].imshow(img_slice, cmap='gray', aspect='auto')
    #         axs[i, 1].imshow(attn_map, cmap='jet', alpha=0.5, aspect='auto')
    #         axs[i, 1].set_title(f'{plane} Plane Attention Map')
    #         axs[i, 1].axis('off')

    #     plt.tight_layout()
    #     plt.savefig('attention_maps_with_original_images.png')
    #     plt.close()

    # else:

    #     #get attention map
    #     model.eval()  # Set the model to evaluation mode
    
    #     # Get a single batch from the DataLoader
    #     inputs, targets = next(iter(train_loader))
        
    #     # Move input to the device (e.g., GPU)
    #     inputs = inputs.to(device)

    #     #pass input through the 3D CNN
    #     threed_convs = model.cnnblock3d(inputs)


    #     #pass convs through to mpms extractor
    #     mpms = model.mpms(threed_convs)

    #     emb = model.embedding_layer(mpms)

    #     #first_attn_block = model.transformer_encoder.layers[0].attention.qkv(emb)[0].squeeze(0)
    #     qkv = model.transformer_encoder.layers[0].attention.qkv(emb)

    #     #specific_first_block_attn = first_attn_block.squeeze(0)

    #     print(qkv.shape)


    #     num_heads = model.transformer_encoder.layers[0].attention.num_heads
    #     emb_size = model.transformer_encoder.layers[0].attention.emb_size
    #     head_dim = emb_size // num_heads  # Dimension per head

    #     qkv = rearrange(
    #         qkv, 'b n (qkv h d) -> qkv b h n d', h=num_heads, qkv=3, d=head_dim
    #     )
    #     queries, keys, values = qkv[0], qkv[1], qkv[2]

    #     # Step 3: Compute attention scores (energy)
    #     energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # Shape: [batch_size, num_heads, seq_len, seq_len]

    #     print(energy.shape)

    #     # Step 4: Apply scaling and softmax
    #     scaling = emb_size ** 0.5
    #     energy = energy / scaling
    #     attention_weights = F.softmax(energy, dim=-1)  # Shape: [batch_size, num_heads, seq_len, seq_len]

    
    #     # Select the first sample
    #     attn_weights_first_sample = attention_weights[0]  # Shape: [num_heads, seq_len, seq_len]

    #     # Average over heads (or select a specific head)
    #     attn_matrix = attn_weights_first_sample.mean(dim=0).detach().cpu().numpy()  # Shape: (88, 88)

    #     # Define token indices for each plane
    #     coronal_indices = list(range(1, 29))   # Positions 1-28
    #     sagittal_indices = list(range(30, 58)) # Positions 30-57
    #     axial_indices = list(range(59, 87))    # Positions 59-86

    #     # Function to visualize attention matrices
    #     def visualize_attention(attn_submatrix, plane_name):
    #         plt.figure(figsize=(8, 6))
    #         plt.imshow(attn_submatrix, cmap='viridis')
    #         plt.colorbar()
    #         plt.title(f'{plane_name} Plane Attention')
    #         plt.xlabel('Key Positions')
    #         plt.ylabel('Query Positions')
    #         plt.tight_layout()
    #         plt.savefig(f'{plane_name.lower().replace(" ", "_")}_attention.png')
    #         plt.close()

    #     # Extract and visualize intra-plane attention
    #     def extract_and_visualize_intra_plane_attention(attn_matrix, indices, plane_name):
    #         attn_submatrix = attn_matrix[np.ix_(indices, indices)]
    #         # Normalize the attention weights
    #         # Sum over keys (columns) for each query (row)
    #         row_sums = attn_submatrix.sum(axis=1, keepdims=True)
            
    #         # Avoid division by zero
    #         row_sums[row_sums == 0] = 1.0
            
    #         # Normalize the submatrix
    #         attn_submatrix_normalized = attn_submatrix / row_sums
    #         visualize_attention(attn_submatrix, f'{plane_name}')

    #     # Coronal Plane
    #     extract_and_visualize_intra_plane_attention(attn_matrix, coronal_indices, 'Coronal')

    #     # Sagittal Plane
    #     extract_and_visualize_intra_plane_attention(attn_matrix, sagittal_indices, 'Sagittal')

    #     # Axial Plane
    #     extract_and_visualize_intra_plane_attention(attn_matrix, axial_indices, 'Axial')

    #     # Extract and visualize inter-plane attention (Optional)
    #     def extract_and_visualize_inter_plane_attention(attn_matrix, query_indices, key_indices, name):
    #         attn_submatrix = attn_matrix[np.ix_(query_indices, key_indices)]
    #         visualize_attention(attn_submatrix, name)

        # Examples of inter-plane attention
        # extract_and_visualize_inter_plane_attention(attn_matrix, coronal_indices, sagittal_indices, 'Coronal to Sagittal')
        # extract_and_visualize_inter_plane_attention(attn_matrix, coronal_indices, axial_indices, 'Coronal to Axial')
        # extract_and_visualize_inter_plane_attention(attn_matrix, sagittal_indices, axial_indices, 'Sagittal to Axial')



        #pass the slices through a 2D pretrained CNN

        #project these slices into higher dimensons

        #pass the embedded volume through the first attenion block and squeeze batch dim as we only use 1 image

        #make the attention map/matrix

        #plot the attention




        
        # Forward pass: model outputs and attention maps
        # with torch.no_grad():  # No need to compute gradients
        #     outputs = model(inputs)
        #     return
        





    optimizer = torch.optim.Adam(model.parameters(), lr=.00001)
    # optimizer = torch.optim.Adam([
    # {'params': model.resnet.parameters(), 'lr': 1e-5},  # Fine-tune pretrained layers slowly
    # {'params': model.mpmsep.parameters(), 'lr': 1e-4},  # Higher LR for new layers
    # {'params': model.el.parameters(), 'lr': 1e-4},
    # {'params': model.te.parameters(), 'lr': 1e-4},
    # {'params': model.ch.parameters(), 'lr': 1e-4}])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
    
    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    global iteration
    iteration = 0

    for epoch in trange(num_epochs):
        
        train_loss = train(model, train_loader, criterion, optimizer, device, writer)
        
        train_metrics = test(model, train_evaluator, train_loader_at_eval, criterion, device, run)
        val_metrics = test(model, val_evaluator, val_loader, criterion, device, run)
        test_metrics = test(model, test_evaluator, test_loader, criterion, device, run)
        
        scheduler.step()
        
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
            
        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)

            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = {
        'net': model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model_latest_new.pth')
    torch.save(state, path)

    if gradcam == 'y' and is_resnet:
        train_metrics = gradcam_generate(best_model, train_evaluator, train_loader_at_eval, criterion, device, run, output_root)
    else:
        train_metrics = test(best_model, train_evaluator, train_loader_at_eval, criterion, device, run, output_root)
        val_metrics = test(best_model, val_evaluator, val_loader, criterion, device, run, output_root)
        test_metrics = test(best_model, test_evaluator, test_loader, criterion, device, run, output_root)

        train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
        val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
        test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

        log = '%s\n' % (data_flag) + train_log + val_log + test_log + '\n'
        print(log)
        
        with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
            f.write(log)        
                
        writer.close()


def train(model, train_loader, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = torch.squeeze(targets, 1).long().to(device)
        loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()
    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, criterion, device, run, save_folder=None):

    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            #outputs, _ = model(inputs.to(device))
            outputs = model(inputs.to(device))

            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())

            y_score = torch.cat((y_score, outputs), 0)

        y_score = y_score.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, save_folder, run)

        test_loss = sum(total_loss) / len(total_loss)

        print(test_loss, auc, acc)
        return [test_loss, auc, acc]
    

#used to generate the Grad-CAM features of a 3d resnet 50 model
def gradcam_generate(model, evaluator, data_loader, criterion, device, run, save_folder=None):

    model.eval()

    model.to(device)

    # Identify the target layer
    target_layer = model.layer4[-1].conv3  # Last conv layer in layer4

    # Initialize dictionaries to store activations and gradients
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    # Register the hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Load your 28x28x28 image
    # Replace with your actual image data
    image_np = np.random.rand(28, 28, 28).astype(np.float32)

    # Prepare the input tensor
    input_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(device)

    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Compute Grad-CAM
    activation = activations['value']
    gradient = gradients['value']
    weights = torch.mean(gradient, dim=(2, 3, 4), keepdim=True)
    grad_cam = torch.sum(weights * activation, dim=1)
    grad_cam = torch.relu(grad_cam)
    grad_cam -= grad_cam.min()
    grad_cam /= grad_cam.max()

    # Upsample Grad-CAM
    grad_cam = F.interpolate(
        grad_cam.unsqueeze(1), size=input_tensor.shape[2:], mode='trilinear', align_corners=False
    ).squeeze(1)

    # Convert tensors to NumPy arrays
    grad_cam_np = grad_cam.squeeze().cpu().numpy()
    input_np = input_tensor.squeeze().cpu().numpy()
    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min())

    # Select slices
    slice_idx = 14  # Middle slice

    axial_image = input_np[:, :, slice_idx]
    axial_cam = grad_cam_np[:, :, slice_idx]

    coronal_image = input_np[:, slice_idx, :]
    coronal_cam = grad_cam_np[:, slice_idx, :]

    sagittal_image = input_np[slice_idx, :, :]
    sagittal_cam = grad_cam_np[slice_idx, :, :]

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(axial_image, cmap='gray')
    axes[0].imshow(axial_cam, cmap='jet', alpha=0.5)
    axes[0].set_title('Axial Plane')
    axes[0].axis('off')

    axes[1].imshow(coronal_image, cmap='gray')
    axes[1].imshow(coronal_cam, cmap='jet', alpha=0.5)
    axes[1].set_title('Coronal Plane')
    axes[1].axis('off')

    axes[2].imshow(sagittal_image, cmap='gray')
    axes[2].imshow(sagittal_cam, cmap='jet', alpha=0.5)
    axes[2].set_title('Sagittal Plane')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{data_flag.lower()}_resnet_attnmap_visualization.png')
    plt.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST3D')

    parser.add_argument('--data_flag',
                        default='fracturemnist3d',
                        help='here you can select the dataset from: fracturemnist3d, adrenalmnist3d, organmnist3d, vesselmnist3d, nodulemnist3d,synapsemnist3d',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=100,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--size',
                        default=28,
                        help='the image size of the dataset, 28 or 64, default=28 as we only utilized the 28 pixel dataset',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=6,
                        type=int)
    parser.add_argument('--conv',
                        default='N',
                        help='choose converter from Conv2_5d, Conv3d, ACSConv, N (leave as N unless using a resnet)',
                        type=str)
    parser.add_argument('--pretrained_3d',
                        default='i3d',
                        type=str)
    parser.add_argument('--download',
                        action="store_false")
    parser.add_argument('--as_rgb',
                        help='to copy channels, tranform shape 1x28x28x28 to 3x28x28x28',
                        action="store_true")
    parser.add_argument('--shape_transform',
                        help='for shape dataset, whether multiply 0.5 at eval',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='3DViTMedNet',
                        help='choose backbone, resnet18/resnet50/vit/3DViTMedNet/medicalnet_transfer/swin',
                        type=str)
    parser.add_argument('--gradcam',
                        default='n',
                        help='selecting if you want to use grad-cam (only works if you have the model flag set to resnet)',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)

    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    size = args.size
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    conv = args.conv
    pretrained_3d = args.pretrained_3d
    download = args.download
    model_flag = args.model_flag
    gradcam = args.gradcam
    as_rgb = args.as_rgb
    model_path = args.model_path
    shape_transform = args.shape_transform
    run = args.run

    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, size, conv, pretrained_3d, download, model_flag, as_rgb, shape_transform, model_path, gradcam, run)
