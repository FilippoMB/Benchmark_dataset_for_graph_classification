import argparse 
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.logging import log
from torch_geometric.nn import MLP, GINConv, global_add_pool

from torch_loader import GraphClassificationBench

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


file_path = "data/"
train_dataset = GraphClassificationBench(file_path, split='train', easy=False, small=False)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
val_dataset = GraphClassificationBench(file_path, split='val', easy=False, small=False)
val_loader = DataLoader(val_dataset, args.batch_size)
test_dataset = GraphClassificationBench(file_path, split='test', easy=False, small=False)
test_loader = DataLoader(test_dataset, args.batch_size)


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)


model = Net(train_dataset.num_features, args.hidden_channels, train_dataset.num_classes,
            args.num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


for epoch in range(1, args.epochs + 1):
    loss = train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
