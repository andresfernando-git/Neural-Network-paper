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
    x_torch = torch.from_numpy(x).float().to(device)  # Ensure input is float32
    v = unflatten(x_torch, all_params)
    Hv_flat = torch.zeros_like(x_torch, dtype=torch.float32)  # Explicitly set to float32
    model.eval()  # Set model to evaluation mode
    for data in train_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        # Compute gradients for this batch's loss
        grads = torch.autograd.grad(loss, all_params, create_graph=True)
        # Compute gradient dot product with vector v
        grad_dot_v = sum((g * v_i).sum() for g, v_i in zip(grads, v))
        # Compute Hessian-vector product
        Hv = torch.autograd.grad(grad_dot_v, all_params, retain_graph=True)
        # Accumulate weighted by batch size
        batch_weight = torch.tensor(len(labels) / len(train_dataset), dtype=torch.float32, device=device)
        Hv_flat += flatten(Hv) * batch_weight  # Use tensor for scaling
    return Hv_flat.cpu().numpy()

# Function to set global random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.sigmoid(self.fc1(x))  # Sigmoid activation
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation
        x = self.fc3(x)  # No activation for final layer (CrossEntropyLoss includes softmax)
        return x

for difsamples in range(1):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = 50
    patience = 5
    batch_size = 64
    learning_rate = 0.0015

    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    set_seed(difsamples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model once per difsamples
    model = Net().to(device)

    # Load reference parameters
    reference_state_dict = torch.load('path/to/reference.pth', map_location=device)
    reference_params = {name: param for name, param in reference_state_dict.items()}

    # Open metrics file and write header
    with open(f'test_data/Model-{difsamples}.csv', 'w') as f:
        f.write('L2_strength,Train_Loss,Train_Reg_Loss,Test_Loss,Test_Accuracy\n')

        # Loop over L2 regularization strengths
        L2_strengths = [i*0.0002 for i in range(501)]
        for L2_strength in L2_strengths:
            # Initialize optimizer with current L2_strength (no weight_decay, as we handle regularization manually)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

            # Early stopping variables
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
                    # Compute L2 regularization: sum of squared differences from reference parameters
                    l2_reg = 0.0
                    for name, param in model.named_parameters():
                        if param.requires_grad and name in reference_params:
                            l2_reg += torch.sum((param - reference_params[name].to(device)) ** 2)
                    effective_loss = loss + L2_strength * l2_reg
                    effective_loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()  # Track unregularized loss for early stopping
                # Compute average training loss
                train_loss = epoch_loss / len(train_loader)

                # Early stopping based on training loss
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    epochs_no_improve = 0
                    best_model_state = model.state_dict()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

            # Restore best model state
            model.load_state_dict(best_model_state)

            # Compute train regularized loss
            train_reg_loss = train_loss
            if L2_strength > 0:
                l2_reg = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and name in reference_params:
                        l2_reg += torch.sum((param - reference_params[name].to(device)) ** 2)
                train_reg_loss += L2_strength * l2_reg.item()

            # Compute test metrics
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            # Compute average test loss and accuracy
            test_loss = test_loss / len(test_loader)
            test_accuracy = test_correct / test_total

            # Compute regularized test loss
            reg_test_loss = test_loss
            if L2_strength > 0:
                l2_reg = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and name in reference_params:
                        l2_reg += torch.sum((param - reference_params[name].to(device)) ** 2)
                reg_test_loss += L2_strength * l2_reg.item()

            # Write metrics to file
            f.write(f'{L2_strength:.5f},{train_loss:.9f},{train_reg_loss:.9f},{test_loss:.9f},{100*test_accuracy:.9f}\n')

            # Compute gradient of the average cross-entropy loss over the training set
            model.train()
            optimizer.zero_grad()
            total_loss = 0.0
            for data in train_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss * (len(labels) / len(train_dataset))

            total_loss.backward()  # Compute gradients of the average loss

            # Save the gradients to a text file
            with open(f'test_data/gradients-{difsamples}-{L2_strength:.5f}.csv', 'w') as grad_file:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_file.write(f'dL/d{name}: ')
                        grad_file.write(', '.join([f'{g:.9f}' for g in param.grad.view(-1).tolist()]))
                        grad_file.write('\n')

            # Save the model's weights and biases
            torch.save(model.state_dict(), f'test_data/model_weights-{difsamples}-{L2_strength:.5f}.pth')

            # Compute Hessian eigenvalues and eigenvectors for the full training loss
            model.eval()
            optimizer.zero_grad()

            # Get all parameters
            all_params = list(model.parameters())

            # Total number of parameters
            num_params = sum(p.numel() for p in all_params)

            # Define the LinearOperator
            hessian_op = LinearOperator((num_params, num_params), matvec=matvec)

            # Compute the top k eigenvalues and eigenvectors
            k = 30
            eigenvalues, eigenvectors = eigsh(hessian_op, k=k, which='LM', tol=1e-3, maxiter=100)
            with open(f'test_data/Hessian-eigenvalues-{difsamples}-{L2_strength:.5f}.csv', 'w') as eigval_file:
                np.savetxt(eigval_file, eigenvalues, delimiter='\n', fmt='%.9f')
            with open(f'test_data/Hessian-eigenvectors-{difsamples}-{L2_strength:.5f}.csv', 'w') as eigvec_file:
                for i in range(eigenvectors.shape[1]): 
                    eigenvector = eigenvectors[:, i]
                    eigvec_file.write(','.join([f'{val:.9f}' for val in eigenvector]))
                    eigvec_file.write('\n')
