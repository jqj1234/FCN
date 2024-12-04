# _*_coding : utf-8 _*_
# @Time : 2024/12/2 20:39
# @Author : jiang
# @File : predict2
# @Project : FCN
import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import fcn_resnet50


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main():
    aux = False  # inference time not need aux_classifier
    classes = 20
    weights_path = "./weights/model_29.pth"
    img_path = "./2007_001311.jpg"
    label_dir = "./data/VOCdevkit/VOC2012/SegmentationClass/"  # Directory containing label images
    palette_path = "./palette.json"

    # Ensure paths exist
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    assert os.path.exists(label_dir), f"label directory {label_dir} not found."

    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = fcn_resnet50(aux=aux, num_classes=classes+1)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    original_img = Image.open(img_path).convert('RGB')
    img_name = os.path.basename(img_path).split('.')[0]+'.png'  # Get the filename from the path
    label_img_path = os.path.join(label_dir, img_name)

    # Ensure the label image exists
    assert os.path.exists(label_img_path), f"label image {label_img_path} not found."

    # Load and resize the label image to match the original image size
    label_img = Image.open(label_img_path).convert('RGB')
    label_img = label_img.resize(original_img.size, Image.NEAREST)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # enter evaluation mode
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)

        # Convert prediction mask to RGB and resize it to match the original image size
        mask = mask.convert('RGB')
        mask = mask.resize(original_img.size, Image.NEAREST)

        # Concatenate images horizontally (left-right)
        total_width = original_img.width * 3
        combined_img = Image.new('RGB', (total_width, original_img.height))
        combined_img.paste(original_img, (0, 0))
        combined_img.paste(label_img, (original_img.width, 0))
        combined_img.paste(mask, (original_img.width * 2, 0))

        # Save the combined image
        combined_img.save("combined_test_result.png")
        print("Combined image saved as 'combined_test_result.png'.")

        # Optionally, also save the mask if needed
        mask.save("test_result.png")


if __name__ == '__main__':
    main()