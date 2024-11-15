# Databricks notebook source
# MAGIC %md
# MAGIC # Setup
# MAGIC
# MAGIC For this notebook to run properly, create a cluster with ML runtime 15.4 LTS. A single node cluster with 60 GBs of memory and a GPU (at least 16GB of VRAM) 

# COMMAND ----------

# MAGIC %pip install mlflow==2.17.2 diffusers==0.31.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow

#set Unity Catalog as the host of MLflow Model Registry
mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# check if cuda available
import torch
torch.cuda.is_available()

# COMMAND ----------

num_cuda_devices = torch.cuda.device_count()
for i in range(num_cuda_devices):
    device_name = torch.cuda.get_device_name(i)
    print(f"CUDA device {i}: {device_name}")

# COMMAND ----------

#log in to huggingface using Databricks managed secrets
from huggingface_hub import login

login(token=dbutils.secrets.get('eo_scope', 'HF_API_TOKEN'))

# COMMAND ----------

#experimenting with the inpainting pipeline
import torch
from diffusers import AutoPipelineForInpainting

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda:0")


# COMMAND ----------

pipe.config

# COMMAND ----------

from PIL import Image

# Load an image from file
image_path = "./raccoon.jpg"
image = Image.open(image_path).convert("RGB")

mask_path = "./raccoon_mask.png"
mask = Image.open(mask_path).convert("L").convert("RGB")

# COMMAND ----------

image.width, image.height, image.width//8*8, image.height//8*8

# COMMAND ----------

assert image.width == mask.width, "Image and mask width should be the same!"
assert image.height == mask.height, "Image and mask height should be the same!"

try:
  assert image.width%8 == 0, "Mask width should be divisible by 8!"
  assert image.height%8 == 0, "Mask height should be divisible by 8!"
except AssertionError:
  image = image.resize((image.width//8*8, image.height//8*8))
  mask = mask.resize((image.width//8*8, image.height//8*8))


# COMMAND ----------

prompt = "bautiful smile, high quality, realistic"
negative_prompt = "bad, ugly, cartoon, low quality"

# prompt = "a realistic head of a cute cat, high quality, realistic"
# negative_prompt = "low quality, cartoon"

#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
output_image = pipe(
  width=image.width,
  height=image.height,
  prompt=prompt, 
  negative_prompt=negative_prompt,
  image=image, 
  mask_image=mask, 
  num_inference_steps=40, 
  guidance_scale=9,
  strength=.81, #.81
).images[0]
output_image.save("./output.png")


# COMMAND ----------

output_image.size

# COMMAND ----------

# MAGIC %md
# MAGIC # MLFlow, UC model registration and serving
# MAGIC In this part we will create our custom MLFlow pyfunc model, log it and then register it to Unity Catalog (UC) to be able to serve it using Databricks model serving.

# COMMAND ----------

from huggingface_hub import snapshot_download

# Download the SDXL inpainting model to a local directory cache
snapshot_location = snapshot_download(repo_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1", local_dir='/local_disk0/sdinpaint/')


# COMMAND ----------

import torch
from diffusers import AutoPipelineForInpainting

from PIL import Image
import io
import base64

import mlflow
mlflow.set_registry_uri('databricks-uc')

class InPainter(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model snapshot directory.
        """
        self.pipe = AutoPipelineForInpainting.from_pretrained(context.artifacts["snapshot"], torch_dtype=torch.float16, variant="fp16")
        
        # NB: If you do not have a CUDA-capable device or have torch installed with CUDA support
        # this setting will not function correctly. Setting device to 'cpu' is valid, but
        # the performance will be very slow.
        # self.pipe.to(device="cpu")
        # If running on a GPU-compatible environment, uncomment the following line:
        self.pipe.to(device="cuda:0")


    def image_to_base64(self, image):
        """
        Convert a PIL Image to binary format using an in-memory bytes buffer.
        
        Parameters:
        image (PIL.Image.Image): The image to convert.

        Returns:
        bytes: The binary representation of the image.
        """
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')  # You can choose a different format if needed
        image_binary = buffer.getvalue()
        img_base64 = base64.b64encode(image_binary)
        img_base64_str = img_base64.decode("utf-8")
        return img_base64_str
    
      
    def base64_to_image(self, img_base64_str):
        # Decode the base64 string to binary
        img_binary = base64.b64decode(img_base64_str)

        # Convert binary data to a BytesIO object
        img_buffered = io.BytesIO(img_binary)

        # Open the image with PIL
        image = Image.open(img_buffered)

        return image


    def predict(self, context, model_input, params=None):
        """
        This method generates prediction for the given input.
        """
        prompt = model_input["prompt"][0]
        negative_prompt = model_input["negative_prompt"][0]
        
        image = self.base64_to_image(model_input['image'][0])
        mask = self.base64_to_image(model_input['mask'][0])

        # Retrieve or use default values for temperature and max_tokens
        num_inference_steps = params.get("num_inference_steps", 30) if params else 30
        guidance_scale = params.get("guidance_scale", 3.5) if params else 3.5
        strength = params.get("strength", .81) if params else .81

        try:
            width = int((image.width/8)*8)
            height = int((image.height/8)*8)
        except Exception:
            width = 1024
            height = 1024

      
        # NB: Sending the tokenized inputs to the GPU here explicitly will not work if your system does not have CUDA support.
        # If attempting to run this with GPU support, change 'cpu' to 'cuda' for maximum performance
        output = self.pipe(
                width=width,
                height=height,
                prompt=prompt, 
                negative_prompt=negative_prompt,
                image=image, 
                mask_image=mask, 
                num_inference_steps=num_inference_steps, 
                guidance_scale=guidance_scale,
                strength=strength
        ).images[0]
        
    
        return {"output_image": self.image_to_base64(output)}

        


# COMMAND ----------

# DBTITLE 1,Creating the model signature
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema

# Define input and output schema
input_schema = Schema(
    [
        ColSpec(DataType.string, "prompt"),
        ColSpec(DataType.string, "negative_prompt"),
        ColSpec(DataType.binary, "image"),
        ColSpec(DataType.binary, "mask")
    ]
)
output_schema = Schema([ColSpec(DataType.binary, "output_image")])

parameters = ParamSchema(
    [
        ParamSpec("num_inference_steps", DataType.integer, np.int32(40), None),
        ParamSpec("max_tokens", DataType.float, np.float32(3.5), None),
        ParamSpec("strength", DataType.float, np.float32(.81), None)
    ]
)

signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=parameters)
signature

# COMMAND ----------

# DBTITLE 1,Logging the model to MLFlow and registering it to UC
with mlflow.start_run(run_name='sdxl_inpaint_run'):
  mlflow.pyfunc.log_model(
        artifact_path="sdxl_inpaint_model",
        python_model=InPainter(),
        artifacts={"snapshot": snapshot_location},
        conda_env='./conda_env.yaml',
        registered_model_name='eo000_ctg.diffusion.sdxl_inpaint',
        signature=signature
  )

# COMMAND ----------

# DBTITLE 1,Serving the model using API (you can also do it in the UI)
import mlflow
from mlflow.deployments import get_deploy_client

ENDPOINT_NAME = 'eo-sdxl-inpainter'

client = get_deploy_client("databricks")

endpoint = client.create_endpoint(
  name=f"{ENDPOINT_NAME}",
  config={
    "served_entities": [
        {
            "entity_name": "eo000_ctg.diffusion.sdxl_inpaint",
            "entity_version": "1",
            "workload_type": "GPU_MEDIUM",
            "workload_size": "Medium",
            "scale_to_zero_enabled": False
        }
    ],
    "traffic_config": {
        "routes": [
            {
                "served_model_name": "sdxl_inpaint-1",
                "traffic_percentage": 100
            }
        ]
    }
  },
  route_optimized=True,
)

