class ProjectionHead(nn.Module):
    def __init__(self, input_dim=44, hidden_dim=16, output_dim=8):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 2. Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # label=1 for same class (positive pair), 0 for different class (negative pair)
        loss = 0.5 * label * torch.pow(distance, 2) + \
