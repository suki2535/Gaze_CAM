import cv2
import numpy as np
import torch
import os
import torchvision.models as models
from torchcam.methods import CAM
from torchvision.transforms import v2
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage
import timm
def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_VIRIDIS,
                      image_weight: float = 0.0) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


import matplotlib.pyplot as plt
transforms = v2.Compose([
        v2.Resize(size=224),
        v2.ToTensor(),
       v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
base_model = 'resnet50d'
version = "ra2_in1k"
model = timm.create_model(f'{base_model}.{version}', pretrained=True, num_classes=3)
# Load a pre-trained model
model.load_state_dict(torch.load('D:/cxr_timm/resnet50d.ra2_in1k.pth', weights_only=True))
model.eval()
# Set up Grad-CAM
cam_extractor = CAM(model, target_layer = 'layer4')  # Use the final convolutional block
#model.train()
# Load an image and preprocess it
#preprocess = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
directory = r"D:\\cxr_timm\\dataset\\test\\Normal"
cnt = 0
l = 0
for filename in os.listdir(directory):
    l += 1
    # Construct full file path
    file_path = os.path.join(directory, filename)
    img1 = Image.open(file_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    #cv2.imshow("xray", img_bgr)
    #cv2.waitKey(0)
    img1 = transforms(img1)
    input_tensor = img1.unsqueeze(0)  # Add batch dimension
    # Forward pass through the model
    output = model(input_tensor)
    # Generate the CAM for the top predicted class
    class_idx = output[0].argmax().item()
    if class_idx == 1:
        cnt += 1
        print(class_idx)
        activation_map = cam_extractor(class_idx, output[0].unsqueeze(0))
        transform = ToPILImage()
        print(activation_map[0][0].size())
        mask = transform(activation_map[0][0])
        print(img_bgr.shape[:2][::-1])
        overlay = mask.resize(img_bgr.shape[:2][::-1], resample=Image.BICUBIC)
        # Convert the activation map to a heatmap and overlay it on the image
        img_bgr = np.array(img_bgr)/255
        mask = np.array(overlay)
        #plt.imshow(mask, cmap='jet')
        result = show_cam_on_image(img_bgr, mask, use_rgb = False)
        '''result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        result = 255 - result
        result = np.array(result)/255
        plt.imshow(result, cmap='jet')'''
        #result = cv2.resize(result , (224,224))
        cv2.imwrite(filename, result)
print(cnt/l)