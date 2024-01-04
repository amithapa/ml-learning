# Create sam predictor
import os
from segment_anything import SamPredictor, sam_model_registry
# wrap it up as function
import base64
import cv2
import numpy as np


model_path = "./sam_vit_b_01ec64.pth"

if not os.path.exists(model_path):
  !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

sam = sam_model_registry["vit_b"](checkpoint=model_path)
predictor = SamPredictor(sam)


def remove_background(image_base4_encoding, x, y):

  image_bytes = base64.b64decode(image_base4_encoding)
  image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

  predictor.set_image(image)

  masks, scores, logits = predictor.predict(
      point_coords=np.asarray([[x, y]]),
      point_labels=np.asarray([1]),
      multimask_output=True
  )

  C, H, W = masks.shape

  result_mask = np.zeros((H, W), dtype=bool)

  for j in range(C):
    result_mask |= masks[j, :, :]

  result_mask = result_mask.astype(np.uint8)

  alpha_channel = np.ones(result_mask.shape, dtype=result_mask.dtype) * 255

  alpha_channel[result_mask == 0] = 0

  result_image = cv2.merge((image, alpha_channel))

  _, result_image_bytes = cv2.imencode(".png", result_image)

  result_image_bytes = result_image_bytes.tobytes()

  encoded_base64_image = base64.b64encode(result_image_bytes).decode("utf-8")

  return encoded_base64_image