import copy
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
import numpy as np
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from collections import OrderedDict
from PIL import Image
import torch.nn.functional as F
from packaging import version
import pytorch_ssim  # or use your own SSIM implementation
from torchmetrics import StructuralSimilarityIndexMeasure


def normalize_for_ssim(x):
    # Convert [-1,1] -> [0,1] for SSIM
    return (x + 1.0) / 2.0


def edge_loss(pred, target):
    # Sobel edge maps
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)

    weight_x = sobel_x.expand(pred.size(1),1,3,3)
    weight_y = sobel_y.expand(pred.size(1),1,3,3)

    edge_pred_x = F.conv2d(pred, weight_x, padding=1, groups=pred.size(1))
    edge_pred_y = F.conv2d(pred, weight_y, padding=1, groups=pred.size(1))
    edge_target_x = F.conv2d(target, weight_x, padding=1, groups=target.size(1))
    edge_target_y = F.conv2d(target, weight_y, padding=1, groups=target.size(1))

    return F.l1_loss(edge_pred_x, edge_target_x) + F.l1_loss(edge_pred_y, edge_target_y)



def overlay_saliency_on_pred(pred, saliency):
    """
    pred: [H, W] or [H, W, 3] image (grayscale or RGB)
    saliency: [H, W] saliency normalized to [0, 1]
    Returns: [H, W, 3] RGB image with saliency overlay
    """
    plt.ioff()

    pred = np.squeeze(pred)
    saliency = np.squeeze(saliency)

    # Normalize saliency map
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # Create plot and overlay
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(pred, cmap='gray')
    ax.imshow(saliency_norm, cmap='jet', alpha=0.5)
    ax.axis("off")
    fig.tight_layout(pad=0)

    # Convert matplotlib figure to image array
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    saliency_overlay = Image.open(buf).convert("RGB")
    saliency_overlay = np.array(saliency_overlay)
    plt.close(fig)

    return saliency_overlay


class ResViT_model(BaseModel):
    def name(self):
        return 'ResViT_model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.psnr_values = []  # Store PSNR for all images
        self.ssim_values = []  # Store SSIM for all images
        
        # Initialize in your model's __init__ or setup
        self.ema_diffmap = None
        self.ema_edge = None
        self.ema_decay = 0.99  # smoothing factor

        # load/define networks
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # pretrained_path = "/home/fnu.talha/resvit/unet_b.ckpt"
        # pretrained_encoder = load_encoder(pretrained_path, in_ch=3, device=device)
        # self.encoder = load_encoder(pretrained_path, in_ch=1, device=device)
        # self.seg_net = load_segmentation_unet(pretrained_path, in_ch=1, device=device)

        # Initialize SSIM metric once (e.g., in __init__ of your model)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        # my changes here
        self.netG = networks.define_G(3, opt.output_nc, opt.ngf,
                                      opt.which_model_netG,opt.vit_name,opt.fineSize,opt.pre_trained_path, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      pre_trained_trans=opt.pre_trained_transformer,pre_trained_resnet = opt.pre_trained_resnet)

        # EMA (Exponential Moving Average) copy of the generator weights.
        # Smooths out generator weight updates over training, which typically
        # gives sharper/more stable samples at test time than the raw weights.
        self.ema_decay_g = getattr(opt, 'ema_decay_g', 0.999)
        self.use_ema = not getattr(opt, 'no_ema', False) and self.ema_decay_g > 0
        if self.use_ema:
            self.netG_ema = copy.deepcopy(self.netG)
            for param in self.netG_ema.parameters():
                param.requires_grad = False
            self.netG_ema.eval()

        # Initialize attributes
        self.saliency = None  # Ensure this attribute always exists

        if self.isTrain:
            self.lambda_f = opt.lambda_f
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,opt.vit_name,opt.fineSize,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
            if self.use_ema:
                try:
                    self.load_network(self.netG_ema, 'G_ema', opt.which_epoch)
                except FileNotFoundError:
                    # No EMA checkpoint available (e.g. resuming a run saved
                    # before EMA was added) - fall back to the raw G weights.
                    self.netG_ema.load_state_dict(self.netG.state_dict())

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'

        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']

        if torch.cuda.is_available() and len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)

        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            input_A = input_A.to(device)
            input_B = input_B.to(device)

        else:
            device = torch.device('cpu')
            input_A = input_A.to(device)
            input_B = input_B.to(device)

        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self, compute_saliency=True):
        # Forward pass through the generator
        self.real_A = Variable(self.input_A)
        # my changes here
        self.fake_B = self.netG(self.real_A[:, 0:3, :, :])  # Generate fake B from real A
        # self.fake_B, self.edge_pred, self.diff_pred = self.netG(self.real_A[:, 0:3, :, :])

        self.real_B = Variable(self.input_B)  # Ground truth B (real B)

        # If we need to compute saliency, do it here
        if compute_saliency:
            # Enable gradient tracking for saliency computation
            self.real_A.requires_grad = True  # Ensure gradients can be computed
            
            # Recompute the fake B with gradient tracking enabled
            self.fake_B = self.netG(self.real_A[:, 0:3, :, :])
            # self.fake_B, self.edge_pred, self.diff_pred = self.netG(self.real_A[:, 0:3, :, :])

            # Compute MSE loss for saliency
            loss = torch.nn.functional.mse_loss(self.fake_B, self.real_B)

            # Compute gradients of the loss with respect to the input (real_A)
            gradients = torch.autograd.grad(outputs=loss, inputs=self.real_A, create_graph=False, retain_graph=True)[0]

            # Compute saliency: absolute gradients for each pixel in the image
            saliency = gradients.abs().mean(dim=1).cpu().numpy().squeeze()  # Mean across channels

            if saliency.max() > 0 and saliency.max() != saliency.min():
                self.saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            else:
                self.saliency = np.zeros_like(saliency)


    def test(self, compute_saliency=False):
        self.saliency = None  # Reset saliency before computation
        torch.cuda.empty_cache()

        # Use the EMA generator for inference when available - its smoothed
        # weights typically give sharper/more stable outputs than the raw G.
        netG = self.netG_ema if self.use_ema else self.netG

        if compute_saliency:
            # Enable gradient tracking
            self.real_A = self.input_A.clone().detach().requires_grad_(True)
            self.real_B = self.input_B
            self.fake_B = netG(self.real_A[:, 0:3, :, :])
            # self.fake_B, self.edge_pred, self.diff_pred = netG(self.real_A[:, 0:3, :, :])

            loss = torch.nn.functional.mse_loss(self.fake_B, self.real_B)
            gradients = torch.autograd.grad(outputs=loss, inputs=self.real_A, create_graph=False, retain_graph=True)[0]
            saliency = gradients.abs().mean(dim=1).cpu().numpy().squeeze()

            if saliency.max() > 0:
                self.saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

        else:
            with torch.no_grad():
                self.real_A = self.input_A
                self.fake_B = netG(self.real_A[:, 0:3, :, :])
                # self.fake_B, self.edge_pred, self.diff_pred = netG(self.real_A[:, 0:3, :, :])
                self.real_B = self.input_B

        # Detach tensors and convert to numpy for visualization
        fake_B_np = self.fake_B.detach().cpu().numpy().squeeze()
        real_B_np = self.real_B.detach().cpu().numpy().squeeze()

        # Normalize from [-1, 1] range to [0, 1]
        fake_B_np = (fake_B_np + 1) / 2
        real_B_np = (real_B_np + 1) / 2

        # Clip to ensure values are strictly in [0, 1]
        fake_B_np = np.clip(fake_B_np, 0, 1)
        real_B_np = np.clip(real_B_np, 0, 1)

        # Handle zero-max cases before PSNR computation
        if real_B_np.max() <= 0 or fake_B_np.max() <= 0:
            return  # Skip this image if there's no valid data

        # Compute PSNR and SSIM
        psnr_value = psnr(real_B_np, fake_B_np, data_range=1.0)
        ssim_value = ssim(real_B_np, fake_B_np, data_range=1.0, multichannel=True)

        # Store values for averaging
        self.psnr_values.append(psnr_value)
        self.ssim_values.append(ssim_value)

        # Compute Difference Map
        diff_map = np.abs(fake_B_np - real_B_np)
        # Normalize difference map to [0, 1] for visualization
        diff_map = np.clip(diff_map / diff_map.max(), 0, 1)

        print(f"Image {len(self.psnr_values)} | PSNR: {psnr_value}, SSIM: {ssim_value}")

        # Return necessary data for visualization and further processing
        return fake_B_np, real_B_np, diff_map, self.saliency


    def compute_final_metrics(self):
        """Compute and print average and standard deviation for PSNR & SSIM after all images are processed."""
        if self.psnr_values and self.ssim_values:
            # Convert lists to PyTorch tensors
            psnr_tensor = torch.tensor(self.psnr_values, dtype=torch.float32)
            ssim_tensor = torch.tensor(self.ssim_values, dtype=torch.float32)

            # Create masks for finite values (excludes `inf` and `NaN`)
            psnr_finite_mask = torch.isfinite(psnr_tensor)
            ssim_finite_mask = torch.isfinite(ssim_tensor)

            # Filter out `inf` and `NaN` values using the masks
            psnr_filtered = psnr_tensor[psnr_finite_mask]
            ssim_filtered = ssim_tensor[ssim_finite_mask]

            # Compute mean and std for PSNR (only on finite values)
            avg_psnr = torch.mean(psnr_filtered) if psnr_filtered.numel() > 0 else torch.tensor(float('nan'))
            std_psnr = torch.std(psnr_filtered) if psnr_filtered.numel() > 0 else torch.tensor(float('nan'))

            # Compute mean and std for SSIM (only on finite values)
            avg_ssim = torch.mean(ssim_filtered) if ssim_filtered.numel() > 0 else torch.tensor(float('nan'))
            std_ssim = torch.std(ssim_filtered) if ssim_filtered.numel() > 0 else torch.tensor(float('nan'))

            print(f"\nFinal Average PSNR: {avg_psnr.item():.3f} ± {std_psnr.item():.3f}, Final Average SSIM: {avg_ssim.item():.3f} ± {std_ssim.item():.3f}\n")

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A[:,0:3,:,:], self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False) #
        # Real
        real_AB = torch.cat((self.real_A[:,0:3,:,:], self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*self.opt.lambda_adv

        self.loss_D.backward()


    def backward_G(self):
        # --- Adversarial loss ---
        fake_AB = torch.cat((self.real_A[:, 0:3, :, :], self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # --- Pixel-wise L1 loss ---
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_G_L1 = self.loss_G_L1 #* 10


        if self.opt.lambda_diffmap > 0:
            # Allow gradient flow through fake_B
            diff_map = torch.abs(self.real_B - self.fake_B)

            # Normalize the difference map (optional)
            # diff_map = diff_map / (diff_map.max() + 1e-8)

            # Compute the auxiliary loss
            self.loss_diffmap = torch.mean(diff_map)
            self.loss_G_diffmap = self.opt.lambda_diffmap * self.loss_diffmap
        else:
            self.loss_G_diffmap = 0

        

        # --- Edge loss ---
        if getattr(self.opt, "lambda_edge", 0) > 0:
            self.loss_G_edge = edge_loss(self.fake_B, self.real_B)
            self.loss_G_edge = self.opt.lambda_edge * self.loss_G_edge
        else:
            self.loss_G_edge = torch.tensor(0.0, device=self.fake_B.device)

        # --- FFT high-pass loss ---
        if getattr(self.opt, "lambda_fft", 0) > 0:
            self.loss_G_fft = self.fft_highpass_loss(self.fake_B, self.real_B, radius_frac=0.06)
            # Optional: multiply by soft diff_map to focus on high-error regions
            self.loss_G_fft = self.opt.lambda_fft * self.loss_G_fft
        else:
            self.loss_G_fft = torch.tensor(0.0, device=self.fake_B.device)

        # --- Combine all losses ---
        self.loss_G = (
            self.loss_G_GAN
            + self.loss_G_L1
            + self.loss_G_diffmap
            + self.loss_G_edge
            + self.loss_G_fft
        )

        self.loss_G.backward()


    # --- FFT high-pass loss function ---
    def fft_highpass_loss(self, pred, target, radius_frac=0.08):
        B,C,H,W = pred.shape
        Fp = torch.fft.fftshift(torch.fft.fft2(pred.squeeze(1)), dim=(-2,-1))
        Ft = torch.fft.fftshift(torch.fft.fft2(target.squeeze(1)), dim=(-2,-1))
        yy, xx = torch.meshgrid(torch.arange(H, device=pred.device), torch.arange(W, device=pred.device))
        cy, cx = H//2, W//2
        r = torch.sqrt((yy-cy).float()**2 + (xx-cx).float()**2)
        r = r / r.max()
        mask = (r > radius_frac).float()

        mag_diff = torch.abs(Fp - Ft)
        masked = mag_diff * mask

        # print("FFT masked mean:", masked.mean().item())  # debug
        return masked.mean()



    @torch.no_grad()
    def update_ema(self):
        """Update the EMA generator: ema = decay * ema + (1 - decay) * G."""
        if not self.use_ema:
            return
        ema_params = dict(self.netG_ema.named_parameters())
        for name, param in self.netG.named_parameters():
            ema_params[name].mul_(self.ema_decay_g).add_(param.data, alpha=1 - self.ema_decay_g)

        ema_buffers = dict(self.netG_ema.named_buffers())
        for name, buffer in self.netG.named_buffers():
            ema_buffers[name].copy_(buffer)

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.update_ema()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('G_diff_map', self.loss_G_diffmap.item()),
                            ('G_FFT', self.loss_G_fft.item()),
                            ('G_edge_loss', self.loss_G_edge.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            ])



    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)

        # Extract T1 and T2
        T1 = real_A[:, :, 0]
        T2 = real_A[:, :, 1]
        Flair = real_A[:, :, 2]

        real_T1 = np.stack([T1, T1, T1], axis=-1)
        real_T2 = np.stack([T2, T2, T2], axis=-1)
        real_Flair = np.stack([Flair,Flair,Flair], axis=-1)

        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)

        diff_map = np.abs(fake_B.astype(np.float32) - real_B.astype(np.float32))
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8) * 255
        diff_map = diff_map.astype(np.uint8)
        
        # Select the first image in the batch for the diff_map
        diff_map = torch.abs(self.real_B[0] - self.fake_B[0]).detach().cpu().numpy().squeeze()  # Selecting first image
        diff_map = diff_map / (diff_map.max() + 1e-8)  # Normalize between 0 and 1
        diff_map = (diff_map * 255).astype(np.uint8)  # Scale to 0-255 range for visualization
            # Convert diff_map to RGB by stacking it as 3 channels
        diff_map = np.stack([diff_map] * 3, axis=-1)

        # visuals['diff_map'] = diff_map

        # Handle saliency overlay visualization
        if self.saliency is None:
            saliency_map_vis = np.zeros_like(fake_B)  # Black placeholder
        else:
            saliency_np = self.saliency
            if saliency_np.ndim == 3:
                saliency_np = saliency_np[0]  # Take the first image from batch
            elif saliency_np.ndim == 4:
                saliency_np = saliency_np[0, 0]  # Handle [B, 1, H, W] case

            saliency_map_vis = overlay_saliency_on_pred(fake_B, saliency_np)


        visuals = OrderedDict([
            ('T1', real_T1),
            ('T2', real_T2),
            ('FLAIR',real_Flair),
            ('real_T1ce', real_B),
            ('Pred_T1ce', fake_B),
            ('Diff_Map', diff_map)
            # ('Saliency_Map', saliency_map_vis)
        ])

        return visuals


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        if self.use_ema:
            self.save_network(self.netG_ema, 'G_ema', label, self.gpu_ids)
        