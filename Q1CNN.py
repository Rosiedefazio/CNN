import os
import argparse
import datetime
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.utils
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)
import glob
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

model_default_args = dict( learning_rate = .01,
    epochs= 10,
    batch_size =32
)
device = "cuda" if torch.cuda.is_available() else "cpu"
wandb_project = "MyProject"


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def get_args_parser():
    parser = argparse.ArgumentParser(description="Training script", add_help=False)
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument("--n_cpu", default=8, type=int)
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--input_dir", default="./data/", type=str)
    parser.add_argument('--compile', default=False, type=boolean_string)
    parser.add_argument(
        "--output_dir",
        default="model_output",
        help="all model outputs will be stored in this dir",
    )
    parser.add_argument("--resume_training", default=False, type=boolean_string)
    parser.add_argument(
        "--saved_model_path", default="./model_output/saved_models", type=str
    )

    return parser

def evaluation_metrics(y_true: np.ndarray, y_preds: np.ndarray):
    conf_matrix = confusion_matrix(y_true, y_preds)
    accuracy, f1, precision, recall = (
        accuracy_score(y_true, y_preds),
        f1_score(y_true, y_preds, zero_division=0.0, average="macro"),
        precision_score(y_true, y_preds, zero_division=0.0, average="macro"),
        recall_score(y_true, y_preds, zero_division=0.0, average="macro"),
    )
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }, conf_matrix


@torch.inference_mode()
def perform_inference(model, dataloader, device: str, loss_fn=None):
    """
    Perform inference on given dataset using given model on the specified device. If loss_fn is provided, it also
    computes the loss and returns [y_preds, y_true, losses].
    """
    model.eval()  # Set the model to evaluation mode, this disables training specific operations such as dropout and batch normalization
    y_preds = []
    y_true = []
    losses = []

    print("[inference.py]: Running inference...")
    for i, batch in tqdm(enumerate(dataloader)):
        inputs = batch["img"].to(device)
        outputs = model(inputs)
        if loss_fn is not None:
            labels = batch["label"].to(device)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            y_true.append(labels.cpu().numpy())

        preds = F.softmax(outputs.detach().cpu(), dim=1).argmax(dim=1)
        y_preds.append(preds.numpy())

    model.train()  # Set the model back to training mode
    y_true, y_preds = np.concatenate(y_true), np.concatenate(y_preds)
    return y_true, y_preds, np.mean(losses) if losses else None

def conf_matrix_plot(cf_matrix: np.ndarray, title: str = ""):
    """
    Return matplotlib fig of confusion matrix
    """
    fig, axs = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
        ax=axs,
    )
    fig.suptitle(title)
    return fig


def evaluation_metrics(y_true: np.ndarray, y_preds: np.ndarray):
    conf_matrix = confusion_matrix(y_true, y_preds)
    accuracy, f1, precision, recall = (
        accuracy_score(y_true, y_preds),
        f1_score(y_true, y_preds, zero_division=0.0, average="macro"),
        precision_score(y_true, y_preds, zero_division=0.0, average="macro"),
        recall_score(y_true, y_preds, zero_division=0.0, average="macro"),
    )
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }, conf_matrix


class BasicCNN1(nn.Module):
    """
    Simple CNN model for RGB images.
    """
    def __init__(self, n_classes, n_blocks, dropout) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2, bias=False),  # 3 input channels for RGB
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.maxpool = nn.MaxPool2d(2, 2)  # Downsamples by factor of 2
        self.features = self.make_feature_layers(n_blocks)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False),  # Input now matches feature output
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Outputs (batch, channels, 1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(32, 32, bias=False),  # Match input to output channels of conv2
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def make_feature_layers(self, n_blocks: int) -> nn.Sequential:
        layers = []
        for _ in range(n_blocks):
            layers += [
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            ]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"Input shape: {x.shape}")
        x = self.maxpool(self.conv1(x))
        print(f"After conv1 and maxpool: {x.shape}")
        x = self.features(x)
        print(f"After feature layers: {x.shape}")
        x = self.avgpool(self.conv2(x))
        print(f"After conv2 and avgpool: {x.shape}")
        x = torch.flatten(x, 1)
        print(f"After flatten: {x.shape}")
        x = self.classifier(x)
        print(f"After classifier: {x.shape}")
        return x

    """
Initialise weights of given module using Kaiming Normal initialisation for linear and convolutional layers, and zeros for bias.
    """
def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model = BasicCNN1(n_classes=3, n_blocks=2, dropout=0.3).to(device)
model = model.apply(init_weights)             # It is generally a good idea to have this done inside the model class itself

source_folder = "./CellCycle"
classes_to_merge = ["G1", "G2", "S"]  # Classes to merge
merged_class_name = "G1_G2_S"  # New name for the merged class
data = []
labels = []
for class_folder in os.listdir(source_folder):
    class_path = os.path.join(source_folder, class_folder)

    if not os.path.isdir(class_path):
         continue

    class_label = merged_class_name if class_folder in classes_to_merge else class_folder
    image_paths = glob.glob(os.path.join(class_path, "*merged*.jpg"))

    data.extend(image_paths)
    labels.extend([class_label] * len(image_paths))


train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, test_size=0.3, stratify=labels, random_state=42
    )
valid_data, test_data, valid_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

print(f"Train set: {len(train_data)} images")
print(f"Validation set: {len(valid_data)} images")
print(f"Test set: {len(test_data)} images")

class CellCycleDataset(Dataset):
    def __init__(self, image_paths, labels, label_to_idx):
        """
        Args:
            image_paths (list): List of file paths to images.
            labels (list): List of corresponding labels for the images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.label_to_idx = label_to_idx

        self.transform = transforms.Compose([
                transforms.ToTensor(),  # Convert image to PyTorch tensor
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label_idx = self.label_to_idx[label]
        return {"img": image, "label": label_idx}

all_labels = sorted(set(train_labels + valid_labels + test_labels))
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
print(f"Label to Index Mapping: {label_to_idx}")

def main(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.output_dir = os.path.join(args.output_dir, time)
    os.makedirs(os.path.join(args.output_dir, "saved_models"), exist_ok=True)

    train_dataset = CellCycleDataset(train_data, train_labels, label_to_idx)
    valid_dataset = CellCycleDataset(valid_data, valid_labels, label_to_idx)
    test_dataset = CellCycleDataset(test_data, test_labels, label_to_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print("[Debug]: Testing DataLoader")
    for batch in train_dataloader:
        print(f"Batch image shape: {batch['img'].shape}")
        print(f"Batch labels: {batch['label']}")
        break   
    n_epochs = 5
    batches_done = 0
    best_loss_val = float("inf")
    epoch_metrics = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10

    class_freqs = np.bincount([sample['label'] for sample in train_dataset]) / len(train_dataset)
    class_freqs = torch.tensor(class_freqs, device=device, dtype=torch.float32)
    model = BasicCNN1(num_classes, n_blocks=2, dropout=0.3).to(device)
    print(model)
    model = torch.compile(model)
    optimiser = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss().to(device)
   
    model.train()
    print(model)
    if args.compile:
        model = torch.compile(model)
    wandb_run_name = (
        f"CNN_lr_{args.learning_rate}_batch_{args.batch_size}"
    )
    wandb.init(project=wandb_project, name=wandb_run_name, config=args)
    print("[train.py]: Starting Training...")

    for epoch in range(n_epochs):
        y_preds = []
        y_train = []
        losses = []
        print(f"[train.py]: Starting epoch {epoch}...")
        for i, data in tqdm(enumerate(train_dataloader)):
            if batches_done == 0:
                    # Log the first batch of images
                    img_grid = torchvision.utils.make_grid(data["img"], nrow=16)
                    wandb.log({"Images from Batch 0": wandb.Image(img_grid)})
                    wandb.watch(model, loss_fn, log="all", log_freq=100)
                    
            inputs, labels = data["img"].to(device), data["label"].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

            preds = F.softmax(outputs, dim=1).argmax(dim=1)
            y_preds.append(preds.cpu().numpy())
            y_train.append(labels.cpu().numpy())
            losses.append(loss.item())

            batches_done += 1

        loss_train = torch.tensor(losses).mean().item()
        y_train, y_preds = np.concatenate(y_train), np.concatenate(y_preds)
        train_metrics, train_conf_matrix = evaluation_metrics(y_train, y_preds)

        y_val, y_preds_val, loss_val = perform_inference(model, valid_dataloader, device, loss_fn)
        val_metrics, val_conf_matrix = evaluation_metrics(y_val, y_preds_val)
    # wandb logging
        train_metrics["Loss"], val_metrics["Loss"] = loss_train, loss_val
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script", parents=[get_args_parser()])
    args = parser.parse_args()
    compile = args.compile
    main(args)