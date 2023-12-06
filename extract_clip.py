import os
import sys
import torch
import clip
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def cv2_to_pil(image):
  """Converts a cv2 image matrix to a PIL image."""
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = Image.fromarray(np.uint8(image))
  return image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
print("Model loaded")

def extract_vid_features(data_path):

    video_features = {}
    for video_id in tqdm(os.listdir(data_path)):
        video_path = os.path.join(data_path, video_id, video_id + '.mp4')
        try:
            cap = cv2.VideoCapture(video_path)

            # Read and save frames
            image_features = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = preprocess(cv2_to_pil(frame)).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_feature = model.encode_image(image)

                image_features.append(image_feature)

            cap.release()

            video_feature = torch.stack(image_features, dim=0).squeeze(1)
            video_features[video_id] = video_feature
        except:
            print(video_id)

    torch.save(video_features, 'clipvitl14_causalvidqa.pth')

def main(args):
    data_path = args[1]
    extract_vid_features(data_path)

if __name__ == "__main__":
    main(sys.argv)
