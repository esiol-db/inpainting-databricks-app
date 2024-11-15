import gradio as gr
import logging
import os
import json
from PIL import Image
import base64
from io import BytesIO
from databricks.sdk import WorkspaceClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Databricks Workspace Client
w = WorkspaceClient()

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."



def base64_to_pil_image(base64_string):
    """
    Converts a Base64-encoded string into a PIL Image.
    """

    # Decode the base64 string into bytes
    image_data = base64.b64decode(base64_string)
    
    # Convert the bytes data to a PIL image
    image = Image.open(BytesIO(image_data))
    
    return image

# Convert the image to binary format in memory
def pil_image_to_base64(pil_image):
    """
    Converts a PIL Image into a Base64-encoded string.
    """

    def add_padding(b64_string):
        while len(b64_string) % 4 != 0:
            b64_string += '='
        return b64_string
    
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # Specify the format (e.g., "PNG", "JPEG")
    image_binary = buffered.getvalue()
    img_base64 = base64.b64encode(image_binary)
    img_base64_str = img_base64.decode("utf8")

    return add_padding(img_base64_str)


def inpaint(image_dict, prompt, negative_prompt, num_inference_steps, guidance_scale, strength):
    # Extract the background image and mask from the ImageEditor output
    image = image_dict["background"]
    mask = image_dict["layers"][0]
    mask = mask.convert("L")
    
    assert image.width == mask.width, "Image and mask width should be the same!"
    assert image.height == mask.height, "Image and mask height should be the same!"

    try:
        assert image.width%8 == 0, "Mask width should be divisible by 8!"
        assert image.height%8 == 0, "Mask height should be divisible by 8!"
    except AssertionError:
        image = image.resize((image.width//8*8, image.height//8*8))
        mask = mask.resize((image.width//8*8, image.height//8*8))
    

    image_base64 = pil_image_to_base64(image.convert("RGB"))
    mask_base64 = pil_image_to_base64(mask.convert("RGB"))
    data = {
        "inputs" : {
            "prompt": prompt, 
            "negative_prompt": negative_prompt,
            "image": image_base64, 
            "mask": mask_base64
            },
        "params" : {
            "num_inference_steps": num_inference_steps, 
            "guidance_scale": guidance_scale, 
            "strength": strength
            }
    }
    json_payload = json.dumps(data)

    # Perform inpainting
    response = w.api_client.do(
        'POST',
        f"/serving-endpoints/{os.getenv('SERVING_ENDPOINT')}/invocations",
        headers={'Content-Type': 'application/json'},
        data=json_payload
    )
    
    base64_string = response['predictions']['output_image']
    output_image = base64_to_pil_image(base64_string)


    return output_image



# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Inpainting")
    
    with gr.Row():
        image_editor = gr.ImageEditor(type="pil",
            label="Upload an image and draw a mask",
            brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
            height=512,
            width=512
        )
        output_image = gr.Image(type='pil', label="Output Image")
    
    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
        negative_prompt = gr.Textbox(label="Negative Prompt")

    with gr.Row():
        num_inference_steps = gr.Slider(
            label='Number of Inference Steps',
            minimum=5,
            maximum=50,
            value=30,
            step=1)
        guidance_scale = gr.Slider(
            label='Guidance Scale',
            minimum=1,
            maximum=10,
            value=3.5,
            step=0.5)
        strength = gr.Slider(
            label='Strength',
            minimum=0,
            maximum=1,
            value=0.81,
            step=0.01)
        

    
    
    generate_btn = gr.Button("Generate")
    

    generate_btn.click(
        fn=inpaint,
        inputs=[image_editor, prompt, negative_prompt, num_inference_steps, guidance_scale, strength],
        outputs=output_image
    )

demo.launch()