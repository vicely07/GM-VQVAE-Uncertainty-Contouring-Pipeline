class KLDivergence(nn.Module):
    "KL divergence between the estimated normal distribution and a prior distribution"
    def __init__(self):
        super(KLDivergence, self).__init__()
        """
        N :  the index N spans all dimensions of input
        N = H x W x D
        """
        self.N = 80*80
    def forward(self, z_mean, z_sigma):
        z_var = z_sigma * 2
        #return (1/self.N) * ( (z_mean**2 + z_var**2 - z_log_var**2 - 1).sum() )
        return 0.5 * ((z_mean**2 + z_var.exp() - z_var - 1).sum())

class L1Loss(nn.Module):

    "Measuring the `Euclidian distance` between prediction and ground truh using `L1 Norm`"
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, x, y):
        N = y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]*y.shape[4]
        return  ( (x - y).abs()).sum() / N

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
