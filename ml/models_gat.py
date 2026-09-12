class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super(GATEncoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr, return_attention=False):
        if return_attention:
            x, (edge_index_1, att1) = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        if return_attention:
            x, (edge_index_2, att2) = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)
        else:
            x = self.conv2(x, edge_index, edge_attr)
        if return_attention:
            return x, (edge_index_2, att2)
        return x

# 2. Edge Reconstructor (Decoder)
class EdgeReconstructor(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(EdgeReconstructor, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * node_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, edge_dim)
        )

    def forward(self, z_u, z_v):
        combined = torch.cat([z_u, z_v], dim=-1)
        return self.mlp(combined)

def run_gat_anomaly_detection():
    print("Loading MPLADS hetero graph...")
    data = torch.load('ml/mplads_hetero_graph.pt', weights_only=False)

    edge_index_hetero = data['mp', 'sanctions', 'ida'].edge_index
    edge_attr_hetero = data['mp', 'sanctions', 'ida'].edge_attr
    mp_nodes_count = data['mp'].num_nodes
    ida_nodes_count = data['ida'].num_nodes
