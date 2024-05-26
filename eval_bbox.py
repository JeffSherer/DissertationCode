import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class BBDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = json.load(open(annotations_file))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"xmlab{self.img_labels[idx]['img_id']}", "source.jpeg")
        image = Image.open(img_name).convert("RGB")
        bounding_box = self.img_labels[idx]["bounding_box"]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(bounding_box, dtype=torch.float)

def evaluate(model_name_or_path, data_path, image_folder, output_path):
    model = LlamaForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = BBDataset(annotations_file=data_path, img_dir=image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    model.eval()
    results = []
    with torch.no_grad():
        for images, bounding_boxes in dataloader:
            outputs = model(images)
            for i in range(len(bounding_boxes)):
                results.append({
                    "image": dataset.img_labels[i]["img_name"],
                    "predicted_bbox": outputs.logits[i].tolist(),
                    "true_bbox": bounding_boxes[i].tolist()
                })

    with open(output_path, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON file with test annotations")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing test images")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the evaluation results")
    args = parser.parse_args()
    evaluate(**vars(args))
