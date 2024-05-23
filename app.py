# python -m venv env
# env\scripts\activate
# py -m streamlit run app.py

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
        return image, image_bytes
    return None, None


# # Function to generate dice image
# def generate_dice_image(image_bytes):
#     if image_bytes:
#         files = {"image": image_bytes}
#         response = requests.post(
#             "http://abdullah0307.pythonanywhere.com/generate_simple_dice_image",
#             files=files,
#         )
#         if response.status_code == 200:
#             dice_image_bytes = response.content
#             dice_image = Image.open(BytesIO(base64.b64decode(dice_image_bytes)))
#             st.image(dice_image, caption="Generated Dice Image", use_column_width=True)
#             return dice_image
#         else:
#             st.error("Failed to fetch the dice image")
#     else:
#         st.warning("Please upload an image first.")
#     return None


# Function to generate dice image
def generate_dice_image(image):
    # response = requests.get("http://127.0.0.1:5000/")
    files = {"image": image}

    # Make the POST request to the API endpoint
    # response = requests.post('https://your-api-endpoint.com/upload', files=files)
    # response = requests.post("http://127.0.0.1:5000/generate_simple_dice_image",files=files)
    response = requests.post(
        "http://geniusdice.pythonanywhere.com/generate_simple_dice_image", files=files
    )
    # print("length of bytes of local server : --------------",len(response1))
    # print("length of bytes of deployed server : --------------",len(response))

    if response.status_code == 200:
        print("response----------------------------------------- ", response)
        response_json = json.loads(response.content)
        # print(response_json,"--------------------------- re")
        if "image" in response_json:
            print("image in response -------------------------")
            dice_image_base64 = response_json["image"]
            image_bytes = base64.b64decode(dice_image_base64)

            image_buffer = BytesIO(image_bytes)
            image = Image.open(image_buffer)

            st.image(image, caption="Generated Dice Image", use_column_width=True)
            return image
        else:
            st.error("Error: 'image' key not found in the response.")
    else:
        st.error(f"Failed to fetch the dice image. Status code: {response.status_code}")


# Function to create download link
def create_download_button(image):
    if image:
        # if st.button("Download Image"):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f"data:file/png;base64,{img_str}"
        st.markdown(
            f'<a href="{href}" download="dice_image.png">Download dice image</a>',
            unsafe_allow_html=True,
        )


# Main function
def main():
    st.title("Dice Image Generator")

    if "dice_image" not in st.session_state:
        st.session_state.dice_image = None

    # Create buttons
    image, image_bytes = open_image()
    if st.button("Generate Dice Image"):
        st.session_state.dice_image = generate_dice_image(image_bytes)

    create_download_button(st.session_state.dice_image)


if __name__ == "__main__":
    main()
