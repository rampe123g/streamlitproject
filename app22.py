# python -m venv env
# env\scripts\activate

import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
import os
import json

# Function to open image file
def open_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = Image.open(BytesIO(image_bytes))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        return image_bytes
    return None


# Function to generate dice image
def generate_dice_image(image):
    # response = requests.get("http://127.0.0.1:5000/")
    files = {'image':image }

    # Make the POST request to the API endpoint
    # response = requests.post('https://your-api-endpoint.com/upload', files=files)
    # response = requests.post("http://127.0.0.1:5000/generate_simple_dice_image",files=files)
    response = requests.post("http://geniusdice.pythonanywhere.com/generate_simple_dice_image",files=files)
    # print("length of bytes of local server : --------------",len(response1))
    # print("length of bytes of deployed server : --------------",len(response))


    if response.status_code == 200:
        print("response----------------------------------------- ",response)
        response_json = json.loads(response.content)
        # print(response_json,"--------------------------- re")
        if 'image' in response_json:
            print("image in response -------------------------")
            dice_image_base64 = response_json['image']
            image_bytes = base64.b64decode(dice_image_base64)
            
            image_buffer = BytesIO(image_bytes)
            image = Image.open(image_buffer)
            
            st.image(image, caption="Generated Dice Image", use_column_width=True)
            return image
        else:
            st.error("Error: 'image' key not found in the response.")
    else:
        st.error(f"Failed to fetch the dice image. Status code: {response.status_code}")


# Function to download image
def download_image(image):
    if image:
        image.save("dice_image.png")
        st.success("Image downloaded successfully!")


# Main function
def main():
    st.title("Image Processing App")

    # Create buttons
    image=None
    dice_image=None
    col1, col2, col3 = st.columns(3)
    with col1:
            image = open_image()
    with col2:
        if image:
            st.image(image,caption="Image")
        if st.button("Generate Dice Image"):
            
            dice_image=generate_dice_image(image)
    with col3:
        if st.button("Download Image"):
            download_image(dice_image)


if __name__ == "__main__":
    main()