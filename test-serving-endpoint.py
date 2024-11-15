# Databricks notebook source
from PIL import Image
import io
import base64
import json

# Load or create a PIL image
image = Image.new("RGB", (1024, 1024), (255, 0, 0))  # Example: a blank red image (height and width should be divisible by 8)
mask = Image.new("RGB", (1024, 1024), (255, 255, 255))  # Example: a blank white image (height and width should be divisible by 8)

def pil_image_to_base64(pil_image):

  def add_padding(b64_string):
    while len(b64_string) % 4 != 0:
        b64_string += '='
    return b64_string
  
  buffered = io.BytesIO()
  pil_image.save(buffered, format="PNG")  # Specify the format (e.g., "PNG", "JPEG")
  image_binary = buffered.getvalue()
  img_base64 = base64.b64encode(image_binary)
  img_base64_str = img_base64.decode("utf8")

  return add_padding(img_base64_str)


def base64_to_pil_image(base64_string):
    # Decode the base64 string into bytes
    image_data = base64.b64decode(base64_string)
    
    # Convert the bytes data to a PIL image
    image = Image.open(io.BytesIO(image_data))
    
    return image

prompt = "Face of a yellow cat"
negative_prompt = "low quality"
image_base64 = pil_image_to_base64(image.convert('RGB'))
mask_base64 = pil_image_to_base64(mask.convert('L').convert('RGB'))

# COMMAND ----------

data = {
  "inputs" : {"prompt": prompt, "negative_prompt": negative_prompt, "image": image_base64, "mask": mask_base64},
  "params" : {"num_inference_steps": 20, "guidance_scale": 7.5, "strength": 1.0}
}

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

ENDPOINT_NAME = "eo-sdxl-inpainter"

response = w.api_client.do(
        'POST',
        f'/serving-endpoints/{ENDPOINT_NAME}/invocations',
        headers={'Content-Type': 'application/json'},
        data=json.dumps(data)
    )
response

# COMMAND ----------

base64_to_pil_image(response["predictions"]["output_image"]).show()

# COMMAND ----------


