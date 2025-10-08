import torch
import torch.nn as nn
import torch.nn.functional as F


def GDLossFuc(a, b):
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
class JacobianDeterminantLoss(nn.Module):
    def __init__(self, epsilon=0.01):
        super(JacobianDeterminantLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, dvf):

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