import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from torch.autograd import Variable
from monai.losses import SSIMLoss
import torch.nn.functional as F
import math
import SimpleITK as sitk

class MorphModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='UNet_3Plus', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')

            parser.add_argument('--ifL1Loss', type=int, default=0, help='use L1Loss for Net_G')
            parser.add_argument('--lambda_L1', type=float, default=1, help='weight for L1 loss')
            parser.add_argument('--ifMSELoss', type=int, default=0, help='use MSELoss for Net_G')
            parser.add_argument('--lambda_MSE', type=float, default=1, help='weight for MSE loss')
            parser.add_argument('--ifMSECTLoss', type=int, default=0, help='use MSECTLoss for Net_G')
            parser.add_argument('--lambda_MSECT', type=float, default=1, help='weight for MSECT loss')
            parser.add_argument('--ifJacobianLoss', type=int, default=0, help='use JacobianLoss for Net_G')
            parser.add_argument('--lambda_Jacobian', type=float, default=1, help='weight for Jacobianloss')

            parser.add_argument('--ifSmoothGDLoss', type=int, default=0, help='use SmoothGDLoss for Net_G')
            parser.add_argument('--lambda_SmoothGD', type=float, default=1, help='weight for SmoothGDloss')
            parser.add_argument('--ifSmoothBELoss', type=int, default=0, help='use SmoothBELoss for Net_G')
            parser.add_argument('--lambda_SmoothBE', type=float, default=1, help='weight for SmoothBEloss')

            parser.add_argument('--ifSSIMCTLoss', type=int, default=0, help='use SSIMCTLoss for Net_G')
            parser.add_argument('--lambda_SSIMCT', type=float, default=1, help='weight for SSIMCT loss')
            parser.add_argument('--ifSSIMDVFLoss', type=int, default=0, help='use SSIMDVFLoss for Net_G')
            parser.add_argument('--lambda_SSIMDVF', type=float, default=1, help='weight for SSIMDVF loss')
            parser.add_argument('--ifGDLoss', type=int, default=0, help='use GDLoss for Net_G')
            parser.add_argument('--lambda_GD', type=float, default=1, help='weight for GD loss')
            parser.add_argument('--ifBCELungLoss', type=int, default=0, help='use BCELungLoss for Net_G')
            parser.add_argument('--lambda_BCELung', type=float, default=1, help='weight for BCELungLoss loss')
            parser.add_argument('--ifBCEGTVLoss', type=int, default=0, help='use BCEGTVLoss for Net_G')
            parser.add_argument('--lambda_BCEGTV', type=float, default=1, help='weight for BCEGTVLoss loss')
            parser.add_argument('--ifDiceLungLoss', type=int, default=0, help='use DiceLungLoss for Net_G')
            parser.add_argument('--lambda_DiceLung', type=float, default=1, help='weight for DiceLung loss')
            parser.add_argument('--ifDiceBodyLoss', type=int, default=0, help='use DiceBodyLoss for Net_G')
            parser.add_argument('--lambda_DiceBody', type=float, default=1, help='weight for DiceBody loss')

            parser.add_argument('--ifProjectionLoss', type=int, default=0, help='use ProjectionLoss for Net_G')
            parser.add_argument('--lambda_Projection', type=float, default=1, help='weight for Projection loss')
            parser.add_argument('--ifMPDLoss', type=int, default=0, help='use MPDLoss for Net_G')
            parser.add_argument('--MPD_P', type=float, default=1.5, help='MPDLoss_P')
            parser.add_argument('--lambda_MPD', type=float, default=80.0, help='weight for MPD loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.GNet = opt.netG
        self.ifGAN = opt.ifGAN
        self.gan_mode = opt.gan_mode
        self.ifattention=opt.ifattention
        self.attention_form = opt.attention_form
        self.spacing=opt.img_spacing
        self.ifL1Loss = opt.ifL1Loss
        self.lambda_L1 = opt.lambda_L1
        self.ifMSELoss = opt.ifMSELoss
        self.lambda_MSE = opt.lambda_MSE
        self.ifMSECTLoss = opt.ifMSECTLoss
        self.lambda_MSECT = opt.lambda_MSECT
        self.ifJacobianLoss = opt.ifJacobianLoss
        self.lambda_Jacobian = opt.lambda_Jacobian
        self.ifSmoothGDLoss = opt.ifSmoothGDLoss
        self.lambda_SmoothGD = opt.lambda_SmoothGD
        self.ifSmoothBELoss = opt.ifSmoothBELoss
        self.lambda_SmoothBE = opt.lambda_SmoothBE
        self.ifSSIMCTLoss = opt.ifSSIMCTLoss
        self.lambda_SSIMCT = opt.lambda_SSIMCT
        self.ifSSIMDVFLoss = opt.ifSSIMDVFLoss
        self.lambda_SSIMDVF = opt.lambda_SSIMDVF
        self.ifGDLoss = opt.ifGDLoss
        self.lambda_GD = opt.lambda_GD
        self.ifBCELungLoss = opt.ifBCELungLoss
        self.lambda_BCELung = opt.lambda_BCELung
        self.ifBCEGTVLoss = opt.ifBCEGTVLoss
        self.lambda_BCEGTV = opt.lambda_BCEGTV
        self.ifDiceLungLoss = opt.ifDiceLungLoss
        self.lambda_DiceLung = opt.lambda_DiceLung
        self.ifDiceBodyLoss = opt.ifDiceBodyLoss
        self.lambda_DiceBody = opt.lambda_DiceBody
        self.ifDiceGTVLoss = opt.ifDiceGTVLoss
        self.lambda_DiceGTV = opt.lambda_DiceGTV
        self.ifMPDLoss = opt.ifMPDLoss
        self.MPD_P = opt.MPD_P
        self.lambda_MPD = opt.lambda_MPD
        self.ifProjectionLoss = opt.ifProjectionLoss
        self.lambda_Projection = opt.lambda_Projection

        self.loss_names = []
        if self.ifL1Loss == 1:
            self.loss_names.append('G_L1Loss')
        if self.ifMSELoss == 1:
            self.loss_names.append('G_MSELoss')
        if self.ifJacobianLoss == 1:
            self.loss_names.append('G_JacobianLoss')
        if self.ifSmoothGDLoss == 1:
            self.loss_names.append('G_SmoothGDLoss')
        if self.ifSmoothBELoss == 1:
            self.loss_names.append('G_SmoothBELoss')
        if self.ifMSECTLoss == 1:
            self.loss_names.append('G_MSECTLoss')
        if self.ifSSIMCTLoss == 1:
            self.loss_names.append('G_SSIMCTLoss')
        if self.ifSSIMDVFLoss == 1:
            self.loss_names.append('G_SSIMDVFLoss')
        if self.ifGDLoss == 1:
            self.loss_names.append("G_GDLoss")
        if self.ifBCELungLoss == 1:
            self.loss_names.append('G_BCELungLoss')
        if self.ifBCEGTVLoss == 1:
            self.loss_names.append('G_BCEGTVLoss')
        if self.ifDiceLungLoss == 1:
            self.loss_names.append('G_DiceLungLoss')
        if self.ifDiceBodyLoss == 1:
            self.loss_names.append('G_DiceBodyLoss')
        if self.ifDiceGTVLoss == 1:
            self.loss_names.append('G_DiceGTVLoss')
        if self.ifProjectionLoss == 1:
            self.loss_names.append('G_ProjectionLoss')
        else:
            pass
        self.visual_names = ["fake_E_1", "real_E_1","fake_E_2", "real_E_2","fake_E_3", "real_E_3","fake_F", "real_F"]

        if self.isTrain and (self.ifGAN == 1):
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, self.ifattention, self.attention_form)

        if self.isTrain and (
                self.ifGAN == 1):
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, 'batch', opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            pass

        if self.isTrain:

            self.L1LossFuc = torch.nn.L1Loss()
            self.MSELossFuc = torch.nn.MSELoss(reduction='mean')
            self.MSECTLossFuc = torch.nn.MSELoss(reduction='mean')
            self.JacobianLossFuc=JacobianDeterminantLoss()
            self.SmoothGDLossFuc = Gradientloss3d()
            self.SmoothBELossFuc=BendingEnergyLoss3D()


            self.SSIMCTLossFuc = SSIMLoss(spatial_dims=3)
            self.SSIMDVFLossFuc = SSIMLoss(spatial_dims=3)

            if (self.ifGAN == 1):
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
                if self.gan_mode == 'wgangp':
                    self.optimizer_D = torch.optim.RMSprop(self.netD.parameters(), lr=opt.lr)
                    self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), lr=opt.lr)
                else:
                    self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
                self.optimizers.append(self.optimizer_G)
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G)

    def set_input(self, input):

        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.real_E_1 = input['E_1'].to(self.device)
        self.real_E_2 = input['E_2'].to(self.device)
        self.real_E_3 = input['E_3'].to(self.device)
        self.real_F = input['F'].to(self.device)
        # self.real_G = input['G'].to(self.device)
        # self.real_H = input['H'].to(self.device)
        self.real_I_1 = input['I_1'].to(self.device)
        self.real_I_2 = input['I_2'].to(self.device)
        self.real_I_3 = input['I_3'].to(self.device)
        self.real_J = input['J'].to(self.device)
        self.real_K = input['K'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        if self.GNet == "Swin_Unetr":
            self.fake_E_1, self.fake_E_2, self.fake_E_3 = nn.Tanh(self.netG(self.real_A,self.real_B,torch.cat((self.real_I_1, self.real_I_2,self.real_I_3), 1)) )
        else:
            self.fake_E_1, self.fake_E_2, self.fake_E_3 = self.netG(torch.cat((self.real_I_1, self.real_I_2,self.real_I_3), 1),self.real_A,self.real_B)


    def backward_D(self):

        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)

        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        if self.ifL1Loss == 1:
            self.loss_G_L1Loss = (self.L1LossFuc(self.fake_E_1, self.real_E_1)+self.L1LossFuc(self.fake_E_2, self.real_E_2)+self.L1LossFuc(self.fake_E_3, self.real_E_3))/3
        else:
            self.loss_G_L1Loss = 0

        if self.ifMSELoss == 1:
            self.loss_G_MSELoss = (self.MSELossFuc(self.fake_E_1, self.real_E_1)+self.MSELossFuc(self.fake_E_2, self.real_E_2)+self.MSELossFuc(self.fake_E_3, self.real_E_3))/3
        else:
            self.loss_G_MSELoss = 0

        if self.ifSSIMDVFLoss == 1:
            self.loss_G_SSIMDVFLoss = (self.SSIMDVFLossFuc(self.fake_E_1, self.real_E_1)+self.SSIMDVFLossFuc(self.fake_E_2, self.real_E_2)+self.SSIMDVFLossFuc(self.fake_E_3, self.real_E_3))/3
        else:
            self.loss_G_SSIMDVFLoss = 0

        if self.ifGDLoss == 1:
            self.loss_G_GDLoss = (self.GDLossFuc(self.fake_E_1, self.real_E_1)+self.GDLossFuc(self.fake_E_2, self.real_E_2)+self.GDLossFuc(self.fake_E_3, self.real_E_3))/3
        else:
            self.loss_G_GDLoss = 0

        self.fake_E = torch.cat([self.fake_E_1, self.fake_E_2, self.fake_E_3], dim=1)
        self.fake_E_scale=self.fake_E*(16)-8


        if self.ifJacobianLoss == 1:
            self.loss_G_JacobianLoss=self.JacobianLossFuc(self.fake_E_scale)
        else:
            self.loss_G_JacobianLoss=0

        if self.ifSmoothGDLoss == 1:
            self.loss_G_SmoothGDLoss=self.SmoothGDLossFuc(self.fake_E_scale)
        else:
            self.loss_G_SmoothGDLoss=0

        if self.ifSmoothBELoss == 1:
            self.loss_G_SmoothBELoss=self.SmoothBELossFuc(self.fake_E_scale)
        else:
            self.loss_G_SmoothBELoss=0

        self.fake_F = warp_volume(self.real_C, self.fake_E_scale, self.spacing, if_lable=False)
        if self.ifSSIMCTLoss == 1:
            self.loss_G_SSIMCTLoss = self.SSIMCTLossFuc(self.fake_F, self.real_F)
        else:
            self.loss_G_SSIMCTLoss = 0

        if self.ifMSECTLoss == 1:
            self.loss_G_MSECTLoss = self.MSECTLossFuc(self.fake_F, self.real_F)
        else:
            self.loss_G_MSECTLoss = 0

        if self.ifProjectionLoss == 1:
            self.loss_G_ProjectionLoss = projection_mse_loss(self.fake_F, self.spacing, self.real_J,SAD=2700.0, SDD=3700.0,det_rows=192,
                                                             det_cols=288,det_height=298.0, det_width=397.0, num_samples=1000, chunk_size=10,angle_deg=0.0)

        else:
            self.loss_G_ProjectionLoss = 0



        if self.ifBCELungLoss == 1:
            self.BCELungLL = nn.BCELoss()
            self.fake_G = warp_volume(self.real_D, self.fake_E_scale, self.spacing, if_lable=True)
            self.loss_G_BCELungLoss = self.BCELungLL(self.fake_G, self.real_G)
        else:
            self.loss_G_BCELungLoss = 0

        if self.ifDiceLungLoss == 1:
            self.fake_G = warp_volume(self.real_D, self.fake_E_scale, self.spacing)
            self.loss_G_DiceLungLoss = self.DiceLoss(self.fake_G, self.real_G)
        else:
            self.loss_G_DiceLungLoss = 0

        if self.ifDiceBodyLoss == 1:
            self.fake_K = warp_volume(self.real_K, self.fake_E_scale, self.spacing)
            self.loss_G_DiceBodyLoss = self.DiceLoss(self.fake_K, self.real_B)
        else:
            self.loss_G_DiceBodyLoss = 0

        self.loss_G = self.loss_G_L1Loss * self.lambda_L1 + self.loss_G_MSELoss * self.lambda_MSE + self.loss_G_JacobianLoss * self.lambda_Jacobian+ self.loss_G_SmoothGDLoss * self.lambda_SmoothGD+ self.loss_G_SmoothBELoss * self.lambda_SmoothBE + self.loss_G_MSECTLoss * self.lambda_MSECT+ self.loss_G_SSIMCTLoss * self.lambda_SSIMCT + self.loss_G_SSIMDVFLoss * self.lambda_SSIMDVF + self.loss_G_GDLoss * self.lambda_GD + self.loss_G_BCELungLoss * self.lambda_BCELung+ self.loss_G_DiceLungLoss * self.lambda_DiceLung+ self.loss_G_DiceBodyLoss * self.lambda_DiceBody+ self.loss_G_ProjectionLoss * self.lambda_Projection
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        if (self.ifGAN == 1):
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            if self.gan_mode == 'wgangp':
                for p in self.netD.parameters():
                    p.data.clamp_(-0.01, 0.01)
            self.set_requires_grad(self.netD, False)
        else:
            pass
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def GDLossFuc(self, a, b):
        GD = 0
        for i in range(a.shape[0]):
            A = a[i][0]
            B = b[i][0]
            if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):

                deltaA_x = torch.abs(A[:(A.shape[0] - 1), :, :] - A[1:, :, :])
                deltaB_x = torch.abs(B[:(B.shape[0] - 1), :, :] - B[1:, :, :])
                deltax = (deltaA_x - deltaB_x) * (deltaA_x - deltaB_x)
                GD_x = deltax.mean()

                deltaA_y = torch.abs(A[:, :(A.shape[1] - 1), :] - A[:, 1:, :])
                deltaB_y = torch.abs(B[:, :(B.shape[1] - 1), :] - B[:, 1:, :])
                deltay = (deltaA_y - deltaB_y) * (deltaA_y - deltaB_y)
                GD_y = deltay.mean()

                deltaA_z = torch.abs(A[:, :, :(A.shape[2] - 1)] - A[:, :, 1:])
                deltaB_z = torch.abs(B[:, :, :(B.shape[2] - 1)] - A[:, :, 1:])
                deltaz = (deltaA_z - deltaB_z) * (deltaA_z - deltaB_z)
                GD_z = deltaz.mean()
                GD += GD_x + GD_y + GD_z
            else:
                raise TypeError("Your input needs to be tensor")
        return GD / a.shape[0]

    def DiceLoss(self, prediction, target):
        smooth = 1e-5
        i_flat = prediction.view(-1)
        t_flat = target.view(-1)

        intersection = (i_flat * t_flat).sum()
        domina = i_flat.sum() + t_flat.sum()
        DSCLoss = 1 - (((2. * intersection) + smooth) / (domina + smooth))
        return DSCLoss



class JacobianDeterminantLoss(nn.Module):
    def __init__(self, epsilon=0.01):
        super(JacobianDeterminantLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, dvf):
        """
        dvf: Tensor of shape [B, 3, D, H, W], representing displacement vector field
        """
        dx = dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :]
        dy = dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :]
        dz = dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1]

        dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
        dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
        dz = F.pad(dz, (0, 1, 0, 0, 0, 0))

        du_dx, du_dy, du_dz = dx[:, 0], dy[:, 0], dz[:, 0]
        dv_dx, dv_dy, dv_dz = dx[:, 1], dy[:, 1], dz[:, 1]
        dw_dx, dw_dy, dw_dz = dx[:, 2], dy[:, 2], dz[:, 2]

        jac_det = (
            du_dx * (dv_dy * dw_dz - dv_dz * dw_dy) -
            du_dy * (dv_dx * dw_dz - dv_dz * dw_dx) +
            du_dz * (dv_dx * dw_dy - dv_dy * dw_dx)
        )

        penalty = F.relu(self.epsilon - jac_det)

        return penalty.mean()


def warp_volume(ct_prior, dvf_mm, spacing,if_lable=True):
    B, _, D, H, W = dvf_mm.shape
    device = dvf_mm.device

    z = torch.linspace(-1, 1, D, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack((xx, yy, zz), dim=0)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)

    spacing_z, spacing_y, spacing_x = spacing[2],spacing[1],spacing[0]

    dvf_voxel = torch.zeros_like(dvf_mm)
    dvf_voxel[:, 0] = dvf_mm[:, 0] / spacing_x
    dvf_voxel[:, 1] = dvf_mm[:, 1] / spacing_y
    dvf_voxel[:, 2] = dvf_mm[:, 2] / spacing_z

    dvf_norm = torch.zeros_like(dvf_voxel)
    dvf_norm[:, 0] = dvf_voxel[:, 0] / ((W - 1) / 2)
    dvf_norm[:, 1] = dvf_voxel[:, 1] / ((H - 1) / 2)
    dvf_norm[:, 2] = dvf_voxel[:, 2] / ((D - 1) / 2)

    warped_grid = (grid + dvf_norm).permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

    if if_lable:
        image_warped = F.grid_sample(ct_prior, warped_grid, mode='nearest', padding_mode='border', align_corners=True)

    else:
        image_warped = F.grid_sample(ct_prior, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)

    return image_warped


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_true, y_pred):
        diff = torch.abs(y_true - y_pred)
        loss = torch.where(diff < self.delta, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta))
        return torch.mean(loss)


class Gradientloss3d(nn.Module):
    def __init__(self, penalty='l2'):
        super(Gradientloss3d, self).__init__()
        assert penalty in ['l1', 'l2']
        self.penalty = penalty

    def forward(self, dvf):
        dx = torch.abs(dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :])
        dy = torch.abs(dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :])
        dz = torch.abs(dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dx = dx ** 2
            dy = dy ** 2
            dz = dz ** 2

        loss = (dx.mean() + dy.mean() + dz.mean()) / 3.0
        return loss

class BendingEnergyLoss3D(nn.Module):
    def __init__(self):
        super(BendingEnergyLoss3D, self).__init__()

    def forward(self, dvf):
        def second_order(f):
            d2x = f[:, 2:, 1:-1, 1:-1] - 2 * f[:, 1:-1, 1:-1, 1:-1] + f[:, :-2, 1:-1, 1:-1]
            d2y = f[:, 1:-1, 2:, 1:-1] - 2 * f[:, 1:-1, 1:-1, 1:-1] + f[:, 1:-1, :-2, 1:-1]
            d2z = f[:, 1:-1, 1:-1, 2:] - 2 * f[:, 1:-1, 1:-1, 1:-1] + f[:, 1:-1, 1:-1, :-2]

            dxy = (
                f[:, 2:, 2:, 1:-1] - f[:, 2:, :-2, 1:-1]
                - f[:, :-2, 2:, 1:-1] + f[:, :-2, :-2, 1:-1]
            ) / 4.0
            dxz = (
                f[:, 2:, 1:-1, 2:] - f[:, 2:, 1:-1, :-2]
                - f[:, :-2, 1:-1, 2:] + f[:, :-2, 1:-1, :-2]
            ) / 4.0
            dyz = (
                f[:, 1:-1, 2:, 2:] - f[:, 1:-1, 2:, :-2]
                - f[:, 1:-1, :-2, 2:] + f[:, 1:-1, :-2, :-2]
            ) / 4.0

            return d2x**2 + d2y**2 + d2z**2 + 2 * (dxy**2 + dxz**2 + dyz**2)

        loss = 0.0
        for i in range(3):
            f = dvf[:, i, :, :, :]
            loss += second_order(f)

        return loss.mean()

def cone_beam_projection(volume, spacing, SAD, SDD,
                         det_rows, det_cols, det_height, det_width,
                         num_samples=2000, chunk_size=1024,
                         angle_deg=0.0):
    volume = volume * 49.95 + 0.02                         
    volume = volume / 1000
    device = volume.device
    D, H, W = volume.shape[1:]
    sx, sy, sz = spacing

    angle = math.radians(angle_deg)

    det_pixel_u = det_width / det_cols
    det_pixel_v = det_height / det_rows

    u = (torch.arange(det_cols, device=device) - det_cols/2 + 0.5) * det_pixel_u
    v = (torch.arange(det_rows, device=device) - det_rows/2 + 0.5) * det_pixel_v
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    vol_center = torch.tensor([W*sx/2, H*sy/2, D*sz/2], device=device)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    R = torch.tensor([[cos_a, -sin_a, 0.0],
                      [sin_a,  cos_a, 0.0],
                      [0.0,    0.0,   1.0]], device=device)

    source_pos0 = torch.tensor([0.0, -SAD, 0.0], device=device)
    det_pos0 = torch.stack([
        uu.flatten(),
        torch.full((det_rows*det_cols,), SDD - SAD, device=device),
        vv.flatten()
    ], dim=-1)

    source_pos = vol_center + (R @ source_pos0)
    det_pos = vol_center + (det_pos0 @ R.T)

    ray_dir = det_pos - source_pos
    ray_dir = ray_dir / torch.norm(ray_dir, dim=-1, keepdim=True)

    t_vals = torch.linspace(0, 2*SAD, num_samples, device=device)
    proj = torch.zeros((det_rows * det_cols,), device=device)
    vol = volume.unsqueeze(0)

    N_rays = det_rows * det_cols
    for start in range(0, N_rays, chunk_size):
        end = min(start + chunk_size, N_rays)
        dir_chunk = ray_dir[start:end]

        ray_points = source_pos[None, None, :] + t_vals[:, None, None] * dir_chunk[None, :, :]

        x_idx = ray_points[..., 0] / sx
        y_idx = ray_points[..., 1] / sy
        z_idx = ray_points[..., 2] / sz

        x_norm = (x_idx / (W-1)) * 2 - 1
        y_norm = (y_idx / (H-1)) * 2 - 1
        z_norm = (z_idx / (D-1)) * 2 - 1

        grid = torch.stack([x_norm, y_norm, z_norm], dim=-1)
        grid = grid.unsqueeze(0).unsqueeze(3)

        mu_vals = F.grid_sample(vol, grid, align_corners=True,
                                mode='bilinear', padding_mode='zeros')
        mu_vals = mu_vals.squeeze()

        dl = torch.norm(dir_chunk, dim=-1) * (t_vals[1]-t_vals[0])
        proj[start:end] = torch.sum(mu_vals, dim=0) * dl

    proj = proj.view(det_rows, det_cols)
    return proj

def projection_mse_loss(volume_all, spacing, xray_gt,
                        SAD=2700.0, SDD=3700.0,
                        det_rows=192, det_cols=288,
                        det_height=298.0, det_width=397.0, num_samples=2000, chunk_size=512,angle_deg=0.0):
    xray_gt_denorm_all = xray_gt * 6
    loss=0
    for i in range(volume_all.shape[0]):
        volume=volume_all[i]
        proj_pred = cone_beam_projection(volume, spacing, SAD, SDD,
                                         det_rows, det_cols,
                                         det_height, det_width,
                                         num_samples,chunk_size,angle_deg)
        proj_pred = proj_pred.flip(0).flip(1)
        pro_np=proj_pred.detach().cpu().numpy()
        pro_img=sitk.GetImageFromArray(pro_np)
        sitk.WriteImage(pro_img,r"test_pro_img.nii")
        xray_gt_denorm=xray_gt_denorm_all[i,0,0,:,:]
        loss+= torch.mean((proj_pred - xray_gt_denorm)**2)
    return loss/volume_all.shape[0]
