import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import VGAE, GCNConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split

# 1. Encoder Architecture
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # Generic 2-layer GCN
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        # Return mu and logvar for the VGAE
        return torch.chunk(x, 2, dim=-1)

def generate_synthetic_elliptic():
    print("Generating synthetic Elliptic-like dataset...")
    # Mimic Elliptic properties
    n_nodes = 10000 # Reduced size for faster validation
    n_features = 147

    # Labels: 0 = licit, 1 = illicit (approx 10% illicit)
    y = np.random.choice([0, 1], size=n_nodes, p=[0.9, 0.1])

    # Features: correlate features with labels
    x = np.random.randn(n_nodes, n_features).astype(np.float32)
    for i in range(n_nodes):
        if y[i] == 1:
            x[i] += 0.5 # Shift mean for illicit nodes

    # Edges: nodes of same label more likely to connect (homophily)
    edge_list = []
    for i in range(n_nodes):
        # Connect to a few random nodes
        for _ in range(3):
            j = np.random.randint(0, n_nodes)
            if i != j:
                # Increase probability if same label
                prob = 0.8 if y[i] == y[j] else 0.2
                if np.random.rand() < prob:
                    edge_list.append([i, j])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

def validate_vgae():
    # Load synthetic dataset
    data = generate_synthetic_elliptic()

    # Split labels for evaluation
    # VGAE is trained on ALL nodes unsupervised.
    indices = np.arange(data.num_nodes)
    train_idx, test_idx = train_test_split(indices, test_size=0.3, stratify=data.y.numpy(), random_state=42)

    # 2. Initialize VGAE
    in_channels = data.num_node_features
    out_channels = 16

    encoder = GCNEncoder(in_channels, out_channels)
    model = VGAE(encoder)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3. Unsupervised Training
    epochs = 100
    print("\nTraining VGAE unsupervised...")
    losses = []

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f}")

    # 4. Evaluation
    model.eval()
    with torch.no_grad():
        mu, _ = model.encoder(data.x, data.edge_index)
        embeddings = mu.numpy()

    # Separate into train/test for the downstream classifier
    X_train = embeddings[train_idx]
    y_train = data.y[train_idx].numpy()
    X_test = embeddings[test_idx]
    y_test = data.y[test_idx].numpy()

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    auprc = average_precision_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)

    # 5. Baseline AUPRC (positive class ratio)
    pos_ratio = np.sum(data.y.numpy() == 1) / len(data.y)

    print("\n" + "="*30)
    print("VGAE Synthetic Validation Results")
    print("="*30)
    print(f"Final Training Loss: {losses[-1]:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Baseline AUPRC (Pos Ratio): {pos_ratio:.4f}")
    print("="*30)

    if auprc <= pos_ratio:
        print("\nWARNING: AUPRC is not meaningfully better than baseline.")
    else:
        print("\nSUCCESS: VGAE embeddings provide discriminative power.")

if __name__ == "__main__":
    validate_vgae()
