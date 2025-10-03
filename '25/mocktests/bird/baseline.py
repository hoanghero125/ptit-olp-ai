import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ---------- Utils ----------
def load_image(path, size=299):
    img = cv2.imread(path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img.astype('float32') / 255.0

def load_data(csv_path, image_dir, has_labels=True):
    df = pd.read_csv(csv_path)
    images, labels, names = [], [], []
    for i, r in df.iterrows():
        p = os.path.join(image_dir, r['image'])
        img = load_image(p)
        if img is None: continue
        images.append(img)
        names.append(r['image'])
        if has_labels: labels.append(r['label'])
        if (i+1) % 200 == 0: print(f"Loaded {i+1}/{len(df)}")
    print("Loaded total:", len(images))
    if has_labels:
        return np.array(images), np.array(labels)
    return np.array(images), np.array(names)

# ---------- Dataset ----------
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(299, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean, std)(img)
        if self.labels is None:
            return img
        return img, int(self.labels[idx])

# ---------- FGSM adversarial ----------
def fgsm_attack(model, images, labels, epsilon=0.01, device='cpu'):
    images = images.clone().detach().to(device)
    images.requires_grad = True
    model.zero_grad()
    outputs = model(images)
    if isinstance(outputs, tuple):
        out, aux = outputs
        loss = F.cross_entropy(out, labels) + 0.4 * F.cross_entropy(aux, labels)
    else:
        loss = F.cross_entropy(outputs, labels)
    loss.backward()
    perturbed = images + epsilon * images.grad.sign()
    perturbed = torch.clamp(perturbed, 0.0, 1.0)
    # renormalize to normalized space if needed (we operate in normalized tensor space already)
    return perturbed.detach()

# ---------- Main training ----------
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Load train CSVs
    X, y = load_data('CUB/train.csv', 'ALL_IMAGES', has_labels=True)
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    n_classes = len(le.classes_); print("Num classes:", n_classes)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

    train_loader = DataLoader(ImageDataset(X_tr, y_tr, transform=train_transform), batch_size=16, shuffle=True, num_workers=2)
    val_loader   = DataLoader(ImageDataset(X_val, y_val, transform=val_transform), batch_size=32, shuffle=False, num_workers=2)

    # Model
    model = models.inception_v3(pretrained=False, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, n_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 10
    epsilon = 0.01  # FGSM strength

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        correct = total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # create adversarial examples (on-the-fly)
            adv = fgsm_attack(model, data, target, epsilon=epsilon, device=device)
            # combine original + adversarial
            inp = torch.cat([data, adv], dim=0)
            tgt = torch.cat([target, target], dim=0)

            optimizer.zero_grad()
            outputs = model(inp)
            if isinstance(outputs, tuple):
                out, aux = outputs
                loss = criterion(out, tgt) + 0.4 * criterion(aux, tgt)
            else:
                out = outputs
                loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == tgt).sum().item()
            total += tgt.size(0)

            if (batch_idx+1) % 20 == 0:
                print(f"Ep[{ep+1}/{epochs}] Batch {batch_idx+1} Loss:{running_loss/(batch_idx+1):.4f} Acc:{100.*correct/total:.2f}%")

        # validation
        model.eval()
        vloss = vcorrect = vtotal = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                if isinstance(outputs, tuple):
                    out, _ = outputs
                else:
                    out = outputs
                vloss += criterion(out, target).item()
                _, preds = torch.max(out, 1)
                vcorrect += (preds == target).sum().item()
                vtotal += target.size(0)
        print(f"Epoch {ep+1} TrainLoss:{running_loss/len(train_loader):.4f} TrainAcc:{100.*correct/total:.2f}% ValLoss:{vloss/len(val_loader):.4f} ValAcc:{100.*vcorrect/vtotal:.2f}%")

    # save
    torch.save(model.state_dict(), 'inceptionv3_fgsm.pth')
    print("Saved model.")

    # Predict test
    X_test, names = load_data('CUB/test.csv', 'ALL_IMAGES', has_labels=False)
    test_loader = DataLoader(ImageDataset(X_test, transform=val_transform), batch_size=32, shuffle=False)
    model.eval()
    preds_all = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            if isinstance(outputs, tuple):
                out, _ = outputs
            else:
                out = outputs
            _, preds = torch.max(out, 1)
            preds_all.extend(preds.cpu().numpy())

    labels = le.inverse_transform(preds_all)
    pd.DataFrame({'image': names, 'label': labels}).to_csv('prediction.csv', index=False)
    print("Predictions saved.")

if __name__ == "__main__":
    train_model()
