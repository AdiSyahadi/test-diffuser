import streamlit as st
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
from PIL import Image

# Load pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    model.to("cuda")
    return model

# Function to generate image
def generate_image(prompt, original_image):
    model = load_model()
    generated_image = model(prompt, image=original_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]
    return generated_image

# Streamlit app
def main():
    st.title("Disney-style 3D Character Generator")
    
    # User input for prompt
    prompt = st.text_input("Enter your prompt:")
    
    # Load original image
    uploaded_image = st.file_uploader("Upload your image:", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        original_image = Image.open(uploaded_image)
        st.image(original_image, caption="Original Image", use_column_width=True)
        
        # Generate and display the image
        if st.button("Generate Character"):
            generated_image = generate_image(prompt, original_image)
            st.image(generated_image, caption="Generated Character", use_column_width=True)

if __name__ == "__main__":
    main()
