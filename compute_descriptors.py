import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from torch.utils.data import Dataset
import numpy as np

import argparse
from pathlib import Path
from PIL import Image


from vpr_model import VPRModel


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_names = []
        self.img_names.extend(img_dir.glob("*.png"))
        self.img_names.extend(img_dir.glob("*.jpg"))
        self.img_names.sort()
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        assert idx >= 0 and idx < len(self.img_names)
        image = Image.open(str(self.img_names[idx]))
        if self.transform:
            image = self.transform(image)
        # returns an image in the proper format
        return image, idx


def set_input_transform(image_size=None):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    if image_size:
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        return T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for batch in tqdm(dataloader, "Calculating descritptors..."):
                imgs, labels = batch
                output = model(imgs.to(device)).cpu()
                print("Labels", labels)
                print("Output shape ", output.shape)
                descriptors.append(np.squeeze(output.numpy()))

    return np.array(descriptors)


def load_model(ckpt_path):
    model = VPRModel(
        backbone_arch="dinov2_vitb14",
        backbone_config={
            "num_trainable_blocks": 4,
            "return_token": True,
            "norm_layer": True,
        },
        agg_arch="SALAD",
        agg_config={
            "num_channels": 768,
            "num_clusters": 64,
            "cluster_dim": 128,
            "token_dim": 256,
        },
    )

    model.load_state_dict(torch.load(ckpt_path))
    model = model.eval()
    model = model.to("cuda")
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model parameters
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        default=None,
        help="Path to the checkpoint",
    )
    parser.add_argument("--img_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)

    parser.add_argument(
        "--image_size", nargs="*", default=None, help="Image size (int, tuple or None)"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")

    args = parser.parse_args()

    # Parse image size
    if args.image_size:
        if len(args.image_size) == 1:
            args.image_size = (args.image_size[0], args.image_size[0])
        elif len(args.image_size) == 2:
            args.image_size = tuple(args.image_size)
        else:
            raise ValueError("Invalid image size, must be int, tuple or None")

        args.image_size = tuple(map(int, args.image_size))

    return args


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model = load_model(args.ckpt_path)

    # Initialize dataset
    input_transform = set_input_transform(args.image_size)
    dataset = ImageDataset(args.img_dir, transform=input_transform)

    dataset_loader = DataLoader(
        dataset,
        num_workers=16,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    descriptors = get_descriptors(model, dataset_loader, "cuda")

    print("Descriptors", descriptors.shape)
    np.savetxt(args.output_file, descriptors)

    print("========> DONE!\n\n")
