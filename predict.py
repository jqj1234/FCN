# _*_coding : utf-8 _*_
# @Time : 2024/12/2 10:41
# @Author : jiang
# @File : predict
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
    aux = False  # 推理不需要使用辅助分类器
    classes = 20
    weights_path = "./weights/model_29.pth"
    img_path = "./test.jpg"
    palette_path = "./palette.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = fcn_resnet50(aux=aux, num_classes=classes + 1)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # 加载权重
    model.load_state_dict(weights_dict)
    model.to(device)

    # 加载图片
    original_img = Image.open(img_path)

    # 归一化
    data_transform = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        # prediction = output['out'].argmax(1).squeeze(0)
        # prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # mask = Image.fromarray(prediction)
        # mask.putpalette(pallette)
        # mask.save("test_result.png")
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)

        # Load the original image and the predicted mask
        original_img = Image.open(img_path).convert('RGB')
        mask = mask.convert('RGB')  # Ensure the mask is in RGB mode for concatenation

        # Resize the mask to match the original image size
        mask = mask.resize(original_img.size, Image.NEAREST)

        # Concatenate images horizontally (left-right)
        combined_img = Image.new('RGB', (original_img.width + mask.width, original_img.height))
        combined_img.paste(original_img, (0, 0))
        combined_img.paste(mask, (original_img.width, 0))

        # Save the combined image
        combined_img.save("combined_test_result.png")
        print("Combined image saved as 'combined_test_result.png'.")

        # Optionally, also save the mask if needed
        mask.save("test_result.png")


if __name__ == '__main__':
    main()
