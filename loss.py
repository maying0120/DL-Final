import torch


class SmoothLoss(torch.nn.Module):

    def __init__(self, smooth=0.0):
        super(SmoothLoss, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction='sum').cuda()
        self.smooth = smooth

    def forward(self, x: torch.Tensor, y: torch.Tensor, norm):
        x = x.contiguous().view(-1, x.size(-1))
        y.contiguous().view(-1)
        trust = x.data.clone()
        trust.fill_(self.smooth / (x.size(1) - 2))
        trust.scatter_(1, y.data.unsqueeze(1), 1 - self.smooth)
        trust[:, 0] = 0
        mask = torch.nonzero(y.data == 0)
        if mask.dim():
            trust.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, torch.autograd.Variable(trust, requires_grad=False)) / norm
