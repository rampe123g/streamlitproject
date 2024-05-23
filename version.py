import base64
import os
from io import BytesIO
from math import ceil

import cv2
import numpy as np
import pygame as pg
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DIE_WIDTH = 25
RESOLUTION_FACTOR = 2

pg.init()

DOT_RADIUS = DIE_WIDTH // 10
GRID_WIDTH = 0  # Will be set dynamically based on input image
GRID_HEIGHT = 0  # Will be set dynamically based on input image

DOT_CENTERS = {
    1: [(0, 0)],
    2: [(1, -1), (-1, 1)],
    3: [(1, -1), (0, 0), (-1, 1)],
    4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
    5: [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0)],
    6: [(-1, -1), (1, -1), (-1, 1), (1, 1), (-1, 0), (1, 0)]
}

DICE_IMAGES = {
    0: cv2.imread(os.getcwd() + "/WhatsApp Image 2024-04-16 at 5.49.44 PM (2).jpeg", 0),
    1: cv2.imread(os.getcwd() + "/WhatsApp Image 2024-04-16 at 5.49.45 PM (1).jpeg", 0),
    2: cv2.imread(os.getcwd() + "/WhatsApp Image 2024-04-16 at 5.49.44 PM (1).jpeg", 0),
    3: cv2.imread(os.getcwd() + "/WhatsApp Image 2024-04-16 at 5.49.45 PM.jpeg", 0),
    4: cv2.imread(os.getcwd() + "/WhatsApp Image 2024-04-16 at 5.49.45 PM (3).jpeg", 0),
    5: cv2.imread(os.getcwd() + "/WhatsApp Image 2024-04-16 at 5.49.45 PM (2).jpeg", 0),
    6: cv2.imread(os.getcwd() + "/WhatsApp Image 2024-04-16 at 5.49.44 PM (2).jpeg", 0),
}

DICE_COLORS = {
    0: (0, 0, 0),       # Black
    1: (0, 0, 255),     # Red
    2: (255, 0, 0),     # Blue
    3: (0, 165, 255),   # Orange
    4: (0, 255, 0),     # Green
    5: (0, 255, 255),   # Yellow
    6: (255, 255, 255)  # White
}

DICE_TEXT_COLORS = {
    0: (255, 255, 255),  # Black
    1: (255, 255, 255),  # Red
    2: (255, 255, 255),  # Blue
    3: (0, 0, 0),        # Orange
    4: (0, 0, 0),        # Green
    5: (0, 0, 0),        # Yellow
    6: (0, 0, 0)         # White
}


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return base64_image


def draw_dice(image):
    global GRID_WIDTH, GRID_HEIGHT

    img = cv2.imread(image, 0)
    HEIGHT, WIDTH = img.shape
    WIDTH *= RESOLUTION_FACTOR
    HEIGHT *= RESOLUTION_FACTOR

    GRID_WIDTH = WIDTH // DIE_WIDTH
    GRID_HEIGHT = HEIGHT // DIE_WIDTH

    w = pg.Surface((GRID_WIDTH * DIE_WIDTH, GRID_HEIGHT * DIE_WIDTH))

    # Downscale image
    img = cv2.resize(img, (GRID_WIDTH, GRID_HEIGHT), interpolation=cv2.INTER_AREA)

    for pixel_x in range(GRID_WIDTH):
        for pixel_y in range(GRID_HEIGHT):
            brightness = img[pixel_y][pixel_x]

            die_number = ceil(brightness / 42.5)  # Map to 1-6

            if die_number == 0:
                continue

            die_x = pixel_x * DIE_WIDTH + 0.5 * DIE_WIDTH
            die_y = pixel_y * DIE_WIDTH + 0.5 * DIE_WIDTH

            for dotCenter in DOT_CENTERS[die_number]:
                dot_x = die_x + dotCenter[0] * DIE_WIDTH * 0.25
                dot_y = die_y + dotCenter[1] * DIE_WIDTH * 0.25
                pg.draw.circle(w, [255, 255, 255], (int(dot_x), int(dot_y)), DOT_RADIUS)

    return w


def draw_lines(surface):
    for i in range(GRID_WIDTH):
        x = i * DIE_WIDTH
        pg.draw.line(surface, [50, 50, 50], (x, 0), (x, GRID_HEIGHT * DIE_WIDTH))

    for i in range(GRID_HEIGHT):
        y = i * DIE_WIDTH
        pg.draw.line(surface, [50, 50, 50], (0, y), (GRID_WIDTH * DIE_WIDTH, y))


@app.route('/generate_simple_dice_image', methods=['POST'])
def generate_simple_dice_image():
    try:
        file = request.files['image']
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes))
        image_path = os.getcwd() + '/input.jpg'
        image = image.convert('RGB')
        image.save(image_path)

        dice_surface = draw_dice(image_path)
        draw_lines(dice_surface)

        image_buffer = pg.image.tostring(dice_surface, 'RGB')
        pil_image = Image.frombytes('RGB', (GRID_WIDTH * DIE_WIDTH, GRID_HEIGHT * DIE_WIDTH), image_buffer)

        response = BytesIO()
        pil_image.save(response, format='PNG')
        response.seek(0)

        # Encode the image bytes in base64
        encoded_image = base64.b64encode(response.getvalue()).decode('utf-8')
        print("image ------------------------------------------------",encoded_image)
        print("Dice Image generated and resturned")

        # Save the encoded image to a local directory (replace with desired path)
        # local_save_path = os.path.join(os.getcwd(), 'output_images', 'dice_image.png')  # Create 'output_images' folder if it doesn't exist
        # os.makedirs(os.path.dirname(local_save_path), exist_ok=True)  # Ensure directory structure exists
        # with open(local_save_path, 'wb') as f:
        #     f.write(base64.b64decode(encoded_image))



        # Return a dictionary as a JSON response with the base64 encoded image
        return encoded_image

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/generate_dice_image_from_dice_image', methods=['POST'])
def generate_dice_image_from_dice_image():
    try:
        # Get the image file and DICE_SIZE from the request
        file = request.files['image']
        dice_size = int(request.form.get('dice_size', 10))
        print(dice_size)

        # Save the image to a temporary file
        image_path = os.getcwd() + '/mysite/input.jpg'
        file.save(image_path)

        # Read and process the image
        img = cv2.imread(image_path, 0)
        HEIGHT, WIDTH = img.shape
        resized = cv2.resize(img, (WIDTH // dice_size, HEIGHT // dice_size))
        divided = resized / 255 * 6
        divided = divided.astype(int)

        for pixel_y in range(divided.shape[0]):
            for pixel_x in range(divided.shape[1]):
                dice = divided[pixel_y, pixel_x]
                dice_image = DICE_IMAGES[dice]
                dice_image = cv2.resize(dice_image, (dice_size, dice_size))
                img[
                pixel_y * dice_size: pixel_y * dice_size + dice_size,
                pixel_x * dice_size: pixel_x * dice_size + dice_size
                ] = dice_image

        # Convert the processed image to base64
        _, buffer = cv2.imencode('.jpg', img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # Return the base64-encoded image in the response
        return jsonify({'image': encoded_image})

    except Exception as e:
        return jsonify({'error': str(e)})


# Flask route to get the result image and HTML as JSON
@app.route('/generate_color_img', methods=['POST'])
def generate_color_img():
    # Get the image file and DICE_SIZE from the request
    file = request.files['image']
    DICE_SIZE = int(request.form.get('dice_size', 25))

    # Save the image to a temporary file
    image_path = os.getcwd() + '/mysite/input.jpg'
    file.save(image_path)

    img = cv2.imread(image_path)

    HEIGHT, WIDTH, _ = img.shape
    resized = cv2.resize(img, (WIDTH // DICE_SIZE, HEIGHT // DICE_SIZE))
    divided = resized / 255 * 7
    divided = divided.astype(int)
    print(np.unique(divided))

    img = cv2.resize(img, (resized.shape[1] * DICE_SIZE, resized.shape[0] * DICE_SIZE))

    for pixel_y in range(divided.shape[0]):
        for pixel_x in range(divided.shape[1]):
            dice_number = divided[pixel_y, pixel_x][0]
            dice_color = tuple(DICE_COLORS[dice_number])
            dice_image = np.ones((DICE_SIZE, DICE_SIZE, 3), dtype=np.uint8) * np.array(dice_color, dtype=np.uint8)
            dice_image = cv2.resize(dice_image, (DICE_SIZE, DICE_SIZE))

            cv2.rectangle(dice_image, (0, 0), (DICE_SIZE, DICE_SIZE), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(dice_image, str(dice_number),
                        (int(DICE_SIZE * 0.1), int(DICE_SIZE * 0.2) * 4),
                        cv2.FONT_HERSHEY_COMPLEX, DICE_SIZE * 0.02, DICE_TEXT_COLORS[dice_number], 1)

            img[pixel_y * DICE_SIZE: pixel_y * DICE_SIZE + DICE_SIZE,
            pixel_x * DICE_SIZE: pixel_x * DICE_SIZE + DICE_SIZE] = dice_image

    # Adding Extra Row
    extra_row = np.ones((DICE_SIZE, img.shape[1], 3), dtype=np.uint8) * 255



    for i in range(extra_row.shape[1] // DICE_SIZE):
        x1 = i * DICE_SIZE
        x2 = x1 + DICE_SIZE

        cv2.rectangle(extra_row[:,x1:x2], (0,0), (DICE_SIZE, DICE_SIZE), (0,0,0), 1,cv2.LINE_AA)
        cv2.putText(extra_row[:, x1:x2], f"{i + 1}",
                    (int(DICE_SIZE * 0.09), int(DICE_SIZE * 0.2) * 4),
                    cv2.FONT_HERSHEY_COMPLEX, DICE_SIZE * 0.015, (0, 0, 0), 1)

    # Adding Extra Column
    extra_col = np.ones((img.shape[0] + DICE_SIZE, DICE_SIZE, 3), dtype=np.uint8) * 255

    rc = cv2.imread(os.getcwd() + "/mysite/rc.jpg")
    rc = cv2.resize(rc, (DICE_SIZE, DICE_SIZE))
    extra_col[:DICE_SIZE,:DICE_SIZE] = rc

    for i in range(1, extra_col.shape[0] // DICE_SIZE):
        y1 = i * DICE_SIZE
        y2 = y1 + DICE_SIZE

        cv2.rectangle(extra_col[y1:y2,:], (0,0), (DICE_SIZE, DICE_SIZE), (0,0,0), 1,cv2.LINE_AA)
        cv2.putText(extra_col[y1:y2, :], f"{i}",
                    (int(DICE_SIZE * 0.09), int(DICE_SIZE * 0.2) * 4),
                    cv2.FONT_HERSHEY_COMPLEX, DICE_SIZE * 0.015, (0, 0, 0), 1)

    row_added = np.concatenate((extra_row, img))
    result_img = np.concatenate((extra_col, row_added), axis=1)

    # Save the result image
    cv2.imwrite(os.getcwd() + '/mysite/result.jpg', result_img)

    # Return the result as JSON
    return jsonify({
        "image": image_to_base64(os.getcwd() + '/mysite/result.jpg'),
        "total_dice": divided.shape[0] * divided.shape[1]
    })

@app.route("/testingap")
def testapp():
    return "appiswoking"

if __name__ == '__main__':
    app.run(debug=True)