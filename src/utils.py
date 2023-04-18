from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision
from vqgan import VQModel
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel, AutoTokenizer

import os
from PIL import Image

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomCrop(256),
])


class ImageCaptionDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.webp']

        self.image_paths = [
            os.path.join(root, filename)
            for root, _, filenames in os.walk(self.dataset_path)
            for filename in filenames
            if os.path.splitext(filename)[1].lower() in self.image_extensions
        ]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with os.fdopen(os.open(image_path, os.O_RDONLY), "rb") as f:
            image = Image.open(f).convert("RGB")
        image = self.transform(image)
        caption = os.path.splitext(os.path.basename(image_path))[0]
        return image, caption


def get_dataloader(dataset_path, batch_size):
    dataset = ImageCaptionDataset(dataset_path)
    return DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)


def load_conditional_models(byt5_model_name, vqgan_path, device):
    vqgan = VQModel().to(device)
    vqgan.load_state_dict(torch.load(
        vqgan_path, map_location=device)['state_dict'])
    vqgan.eval().requires_grad_(False)

    byt5 = T5EncoderModel.from_pretrained(byt5_model_name).to(
        device).eval().requires_grad_(False)
    byt5_tokenizer = AutoTokenizer.from_pretrained(byt5_model_name)

    return vqgan, (byt5_tokenizer, byt5)


def sample(model, model_inputs, latent_shape, unconditional_inputs=None, steps=12, renoise_steps=11, temperature=(1.0, 0.2), cfg=8.0, t_start=1.0, t_end=0.0, device="cuda"):
    with torch.inference_mode():
        sampled = torch.randint(0, model.num_labels,
                                size=latent_shape, device=device)
        init_noise = sampled.clone()
        t_list = torch.linspace(t_start, t_end, steps+1)
        temperatures = torch.linspace(temperature[0], temperature[1], steps)
        for i, t in enumerate(t_list[:steps]):
            t = torch.ones(latent_shape[0], device=device) * t

            logits = model(sampled, t, **model_inputs)
            if cfg:
                logits = logits * cfg + \
                    model(sampled, t, **unconditional_inputs) * (1-cfg)
            scores = logits.div(temperatures[i]).softmax(dim=1)

            sampled = scores.permute(0, 2, 3, 1).reshape(-1, logits.size(1))
            sampled = torch.multinomial(sampled, 1)[:, 0].view(
                logits.size(0), *logits.shape[2:])

            if i < renoise_steps:
                t_next = torch.ones(
                    latent_shape[0], device=device) * t_list[i+1]
                sampled = model.add_noise(
                    sampled, t_next, random_x=init_noise)[0]
    return sampled
