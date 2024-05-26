import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class BBDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = json.load(open(annotations_file))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx]["img_name"])
        image = Image.open(img_path).convert("RGB")
        bounding_box = self.img_labels[idx]["bounding_box"]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(bounding_box, dtype=torch.float)

def train(model_name_or_path, data_path, image_folder, output_dir, num_train_epochs):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = BBDataset(annotations_file=data_path, img_dir=image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.SmoothL1Loss()

    model.train()
    for epoch in range(num_train_epochs):
        for images, bounding_boxes in dataloader:
            outputs = model(images)
            loss = criterion(outputs.logits, bounding_boxes)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{num_train_epochs}, Loss: {loss.item()}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    args = parser.parse_args()
    train(**vars(args))
