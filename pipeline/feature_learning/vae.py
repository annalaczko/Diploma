"""Concrete feature learning steps."""

import csv
import math
from typing import Literal
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import dataloader
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from pipeline.feature_learning.featurelearning_step import FeatureLearningStep

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class ClassificationDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = list(paths)
        self.labels = torch.tensor(labels, dtype=torch.long)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            image = image.convert("L")
            image = self.transform(image)

        return image, self.labels[idx]


class VAEClassifier(nn.Module, FeatureLearningStep):
    def __init__(
        self,
        vae,
        num_classes=28,
        labels_path=None,
        freeze_encoder=False,
        output_path=None,
        name=None,
        delete_input_data=False,
        save_results=False,
        run_id="default_run",
        input_path=None,
    ):
        nn.Module.__init__(self)

        FeatureLearningStep.__init__(
            self,
            input_path=input_path or vae.input_path,
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            save_results=save_results,
            run_id=run_id
        )

        self.encoder = vae.encoder
        self.labels_path = labels_path
        self.freeze_encoder = freeze_encoder

        latent_size = self.encoder.fc_mu.out_features

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        hidden_size = max(latent_size // 2, 32)
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_size),
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        mu, _ = self.encoder(x)
        return self.classifier(mu)

    def run(
        self,
        epochs=30,
        lr=1e-4,
        batch_size=64,
        validation_size=0.2,
        num_workers=0,
    ) -> pd.DataFrame:

        # Checking label path correctness and reading labels

        if not self.labels_path:
            raise ValueError("labels_path is required")

        labels_df = pd.read_csv(self.labels_path)
        labels_df["ID"] = labels_df["ID"].astype(str)

        # Checking that the number of classes is correct and encoding labels
        encoder = LabelEncoder()
        labels_df["label"] = encoder.fit_transform(
            labels_df["TypeRev"].astype(str)
        )

        if len(encoder.classes_) != 28:
            raise ValueError(
                f"Expected 28 classes, got {len(encoder.classes_)}"
            )

        # Creating a mapping from image IDs to labels and filtering the dataset to include only labelled images

        labels_by_id = dict(
            zip(labels_df["ID"], labels_df["label"])
        )

        # Filtering the dataset to include only labelled images

        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = sorted(
            path
            for path in Path(self.input_path).rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
            and path.stem in labels_by_id
        )

        

        print(f"Found {len(paths)} labelled images in {self.input_path}")

        if not paths:
            raise ValueError("No labelled images found")

        labels = [labels_by_id[path.stem] for path in paths]

        label_counts = pd.Series(labels).value_counts()
        valid_labels = label_counts[label_counts >= 2].index
        valid_labels = set(valid_labels.tolist())

        filtered_data = [
            (path, label)
            for path, label in zip(paths, labels)
            if label in valid_labels
        ]
        if not filtered_data:
            raise ValueError("No classes with at least two labelled images found")

        paths, labels = zip(*filtered_data)
        paths = list(paths)

        label_order = sorted(valid_labels)
        label_mapping = {
            label: index for index, label in enumerate(label_order)
        }
        labels = [label_mapping[label] for label in labels]

        classes = encoder.classes_[label_order]
        print(
            f"Using {len(classes)} classes and {len(paths)} labelled images "
            "after excluding singleton classes"
        )
        classifier_device = next(self.parameters()).device
        self.classifier[-1] = nn.Linear(
            self.classifier[-1].in_features,
            len(classes),
        ).to(classifier_device)

        

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths,
            labels,
            test_size=validation_size,
            random_state=42,
            stratify=labels,
        )

        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }

        train_loader = torch.utils.data.DataLoader(
            ClassificationDataset(train_paths, train_labels),
            shuffle=True,
            **loader_kwargs,
        )

        val_loader = torch.utils.data.DataLoader(
            ClassificationDataset(val_paths, val_labels),
            shuffle=False,
            **loader_kwargs,
        )

        device = next(self.parameters()).device
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr,
        )

        history = []

        for epoch in range(epochs):
            self.train()

            if self.freeze_encoder:
                self.encoder.eval()

            train_loss = 0.0
            train_correct = 0
            train_count = 0

            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad()

                logits = self(images)
                loss = criterion(logits, targets)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                train_correct += (
                    logits.argmax(dim=1) == targets
                ).sum().item()
                train_count += images.size(0)

            self.eval()
            val_correct = 0
            val_count = 0

            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    logits = self(images)

                    val_correct += (
                        logits.argmax(dim=1) == targets
                    ).sum().item()
                    val_count += images.size(0)

            train_acc = train_correct / train_count
            val_acc = val_correct / val_count
            average_loss = train_loss / train_count

            history.append({
                "epoch": epoch + 1,
                "loss": average_loss,
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
            })

            print(
                f"[{epoch + 1}/{epochs}] "
                f"loss={average_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

        output_dir = Path(self.output_path or "classifier_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "classes": classes.tolist(),
            },
            output_dir / "vae_classifier.pt",
        )

        history_df = pd.DataFrame(history)
        history_df.to_csv(output_dir / "training_history.csv", index=False)

        return history_df

class VAE(nn.Module, FeatureLearningStep):


    def __init__(
        self,
        input_path: str,
        f=4,
        n=3,
        s=1,
        k=128,
        output_path=None,
        name=None,
        delete_input_data=False,
        save_results=False,
        run_id="default_run",
    ):
        nn.Module.__init__(self)
        FeatureLearningStep.__init__(
            self,
            input_path=input_path or "",
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            save_results=save_results,
            run_id=run_id,
        )
        self.encoder = Encoder(f, n, s, k)
        self.decoder = Decoder(f, n, s, k)

    def run(
        self,
        mode: Literal["extract_features", "train_model"] = "extract_features",
        *,
        epochs=10,
        lr=1e-6,
        batch_size=32
    ) -> pd.DataFrame:
        
        if mode not in {"extract_features", "train_model"}:
            raise ValueError(
                "mode must be either 'extract_features' or 'train_model'"
            )

        folder = self.input_path
        if not folder:
            raise ValueError("input_path is required to run the VAE feature step")

        if mode == "train_model":
            return self.train_model(
                folder,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )

        return self.extract_features(folder, batch_size=batch_size)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = self.decoder.fc(z)
        z = z.view(-1, 128, 32, 32)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, sigma
    
    def train_model(self, folder, epochs=10, lr=1e-6, batch_size=32):
        device = next(self.parameters()).device
        dataset = ImageFolderDataset(folder)
        print(f"Dataset size: {dataset.__len__()}")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=16, pin_memory=True, persistent_workers=True
        )
        optimizer = optim.Adam(self.parameters(), lr=lr)

        self.train()
        for epoch in range(epochs):
            train_loss = 0.0
            beta = self._cyclical_beta(epoch+1, epochs, 4)
            for i, inputs in enumerate(loader):
                inputs = inputs.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs, mu, sigma = self(inputs)
                loss = self._loss_function(outputs, inputs, mu, sigma, beta)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f'[epoch {epoch+1}] loss: {train_loss / len(loader):.3f} | beta: {beta:.3f}')

        return self.extract_features(batch_size=batch_size)

    def extract_features(self, batch_size=32) -> pd.DataFrame:
        """Extract latent vectors and optionally save them with filenames as CSV.

        Returns a DataFrame containing one filename and feature vector per row.
        """
        device = next(self.parameters()).device
        dataset = ImageFolderDataset(self.input_path)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=16, pin_memory=True, persistent_workers=True
        )

        filenames = [Path(p).name for p in dataset.paths]
        self.eval()
        all_features = []
        with torch.no_grad():
            for inputs in loader:
                inputs = inputs.to(device, non_blocking=True)
                mu, _ = self.encoder(inputs)
                all_features.append(mu.cpu())

        features = torch.cat(all_features, dim=0).numpy()

        result = pd.DataFrame({
            "filename": filenames,
            "features": list(features),
        })
        output_file = Path(self.output_path) / "features.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_file, index=False)
        return result

    @staticmethod
    def _loss_function(recon_x, x, mu, logvar, beta=1.0):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD

    @staticmethod
    def _cyclical_beta(epoch, T, M, R=0.5):
        """
        epoch: current epoch (1-indexed, matches 'e' in the paper)
        T: total number of epochs
        M: number of cycles
        R: proportion of each cycle spent annealing (fraction that ramps 0->1)
        """
        period = T / M
        tau = ((epoch - 1) % math.ceil(period)) / period
        if tau <= R:
            return tau / R          # linear ramp 0 -> 1
        else:
            return 1.0              # fixed stage, beta = 1


class Encoder(nn.Module):
    def __init__(self, f=4, n=3, s=1, k=128, p=1):
        super(Encoder, self).__init__()

        #used multiple times
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.25)

        #block0
        self.conv0 = nn.Conv2d(1, 2**f, kernel_size=n, stride=s, padding=p)

        #block1
        self.conv1_0 = nn.Conv2d(2**f, 2**(f+1), kernel_size=n, stride=s+1, padding=p)
        self.conv1_1 = nn.Conv2d(2**(f+1), 2**(f+1), kernel_size=n, stride=s, padding=p)

        #block2
        self.conv2_0 = nn.Conv2d(2**(f+1), 2**(f+2), kernel_size=n, stride=s+1, padding=p)
        self.conv2_1 = nn.Conv2d(2**(f+2), 2**(f+2), kernel_size=n, stride=s, padding=p)

        #block3
        self.conv3_0 = nn.Conv2d(2**(f+2), 2**(f+3), kernel_size=n, stride=s+1, padding=p)
        self.conv3_1 = nn.Conv2d(2**(f+3), 2**(f+3), kernel_size=n, stride=s, padding=p)

        #block4
        flatten_size = 131072

        self.fc_mu = nn.Linear(flatten_size, k)
        self.fc_sigma = nn.Linear(flatten_size, k)

    def forward(self, x):
        #block0
        x = self.conv0(x)
            
        #block1
        x = self.elu(self.conv1_0(x))
        x = self.elu(self.conv1_1(x))
        x = self.dropout(x)

        #block2
        x = self.elu(self.conv2_0(x))
        x = self.elu(self.conv2_1(x))
        x = self.dropout(x)

        #block3
        x = self.elu(self.conv3_0(x))
        x = self.elu(self.conv3_1(x))
        x = self.dropout(x)

        #block4
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)

        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, f=4, n=3, s=1, k=128, p=1):
        super(Decoder, self).__init__()

        #used multiple times
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.25)

        #block0
        self.fc = nn.Linear(k, 2**(f+3) * 32 * 32)
        self.conv0 = nn.ConvTranspose2d(k, 2**(f+3), kernel_size=n, stride=s, padding=p)

        #block1
        self.conv1_0 = nn.ConvTranspose2d(2**(f+3), 2**(f+3), kernel_size=n, stride=s, padding=p)
        self.conv1_1 = nn.ConvTranspose2d(2**(f+3), 2**(f+2), kernel_size=n, stride=s+1, padding=p, output_padding=1)

        #block2
        self.conv2_0 = nn.ConvTranspose2d(2**(f+2), 2**(f+2), kernel_size=n, stride=s, padding=p)
        self.conv2_1 = nn.ConvTranspose2d(2**(f+2), 2**(f+1), kernel_size=n, stride=s+1, padding=p, output_padding=1)

        #block3
        self.conv3_0 = nn.ConvTranspose2d(2**(f+1), 2**(f+1), kernel_size=n, stride=s, padding=p)
        self.conv3_1 = nn.ConvTranspose2d(2**(f+1), 2**(f), kernel_size=n, stride=s+1, padding=p, output_padding=1)

        #block4
        self.conv4 = nn.ConvTranspose2d(2**f, 1, kernel_size=n, stride=s, padding=p)

    def forward(self, x):
        #block0
        x = self.conv0(x)
            
        #block1
        x = self.elu(self.conv1_0(x))
        x = self.elu(self.conv1_1(x))
        x = self.dropout(x)

        #block2
        x = self.elu(self.conv2_0(x))
        x = self.elu(self.conv2_1(x))
        x = self.dropout(x)

        #block3
        x = self.elu(self.conv3_0(x))
        x = self.elu(self.conv3_1(x))
        x = self.dropout(x)

        #block4
        x = self.conv4(x)

        return torch.sigmoid(x)

def resize_with_padding(img, target_size=256, fill=255):
    img = img.copy()
    img.thumbnail((target_size, target_size))
    delta_w = target_size - img.size[0]
    delta_h = target_size - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding, fill=fill)


class ImageFolderDataset(Dataset):
    def __init__(self, folder, target_size=256):
        extensions = {".jpg", ".jpeg", ".png"}
        self.paths = sorted(
            path
            for path in Path(folder).rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        return self.transform(img)