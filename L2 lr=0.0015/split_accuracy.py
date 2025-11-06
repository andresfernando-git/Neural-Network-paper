import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

def flatten(params):
    return torch.cat([p.view(-1) for p in params])

def unflatten(flat, params_template):
    sizes = [p.numel() for p in params_template]
    shapes = [p.shape for p in params_template]
    parts = torch.split(flat, sizes)
    return [part.view(shape) for part, shape in zip(parts, shapes)]

def matvec(x):
    x_torch = torch.from_numpy(x).float().to(device)
    v = unflatten(x_torch, all_params)
    Hv_flat = torch.zeros_like(x_torch, dtype=torch.float32)
    model.eval()
    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        grads = torch.autograd.grad(loss, all_params, create_graph=True)
        grad_dot_v = sum((g * v_i).sum() for g, v_i in zip(grads, v))
        Hv = torch.autograd.grad(grad_dot_v, all_params, retain_graph=True)
        batch_weight = torch.tensor(len(labels) / len(train_dataset), dtype=torch.float32, device=device)
        Hv_flat += flatten(Hv) * batch_weight
    return Hv_flat.cpu().numpy()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

for difsamples in range(1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = 50
    patience = 5
    batch_size = 64
    learning_rate = 0.0015

    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    set_seed(difsamples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = Net().to(device)
    all_params = list(model.parameters())

    # Open metrics file and write header (include per-class accuracies)
    with open(f'Model-{difsamples}_split_accuracy.csv', 'w') as f:
        header = 'L2_strength,Train_Loss,Train_Reg_Loss,Test_Loss,Test_Accuracy,' + ','.join([f'Class_{i}_Accuracy' for i in range(10)]) + '\n'
        f.write(header)

        # Loop over L2 regularization strengths
        L2_strengths = [i*0.0002 for i in range(451)]
        for L2_strength in L2_strengths:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_strength)
            best_train_loss = float('inf')
            epochs_no_improve = 0
            best_model_state = None

            # Training loop
            for epoch in range(1, num_epochs + 1):
                model.train()
                epoch_loss = 0.0
                for data in train_loader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                train_loss = epoch_loss / len(train_loader)

                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    epochs_no_improve = 0
                    best_model_state = model.state_dict()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch} epochs for L2_strength: {L2_strength}")
                    break

            model.load_state_dict(best_model_state)
            train_reg_loss = train_loss
            if L2_strength > 0:
                l2_norm = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
                train_reg_loss += L2_strength * l2_norm.item()

            # Compute test metrics and per-class accuracies
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            class_correct = torch.zeros(10, device=device)
            class_total = torch.zeros(10, device=device)

            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    # Per-class accuracy
                    for i in range(10):
                        class_mask = (labels == i)
                        class_correct[i] += (predicted[class_mask] == labels[class_mask]).sum().item()
                        class_total[i] += class_mask.sum().item()

            test_loss = test_loss / len(test_loader)
            test_accuracy = test_correct / test_total
            reg_test_loss = test_loss
            if L2_strength > 0:
                l2_norm = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
                reg_test_loss += L2_strength * l2_norm.item()

            # Compute per-class accuracies
            class_accuracies = [class_correct[i].item() / class_total[i].item() if class_total[i] > 0 else 0 for i in range(10)]

            # Write metrics to file
            f.write(f'{L2_strength:.5f},{train_loss:.9f},{train_reg_loss:.9f},{test_loss:.9f},{100*test_accuracy:.9f}')
            for acc in class_accuracies:
                f.write(f',{100*acc:.9f}')
            f.write('\n')
