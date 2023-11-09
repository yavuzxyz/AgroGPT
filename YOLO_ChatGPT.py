# This file is part of AgroGPT, which is released under the GNU General Public License (GPL).
# See file LICENSE or go to https://www.gnu.org/licenses/gpl-3.0.html for full license details.

import cv2
import torch
import ultralytics
import numpy as np
from collections import Counter
from ultralytics import YOLO
import time

# Load your custom trained models for segmentation and detection
seg_model = YOLO("C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/Snail_segment/segment.pt")
det_model = YOLO("C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/Snail_segment/detect.pt")

# Image file to run predictions on
image_path = "C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/Snail_segment/Picture1.jpg"

# Load the image
image = cv2.imread(image_path)

# Record the start time
# start_time = time.time()

# Run predictions with a confidence threshold of 0.1 for segmentation
seg_results = seg_model.predict(image, conf=0.1)
print(seg_results)
# Get the predicted masks
masks = seg_results[0].masks

# Create a copy of the original image to apply masks
combined_image = np.zeros_like(image)  # Initially the combined image is all black

# For each predicted result
for result in seg_results:
    masks = result.masks
    boxes = result.boxes
    class_names = result.names

    # Create a copy of the original image to apply masks
    combined_image = np.zeros_like(image)  # Initially the combined image is all black

    # For each predicted mask
    for i, mask in enumerate(masks):
        # Convert the mask to a numpy array if it's not already
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().detach().numpy()
        elif isinstance(mask, ultralytics.yolo.engine.results.Masks):
            mask = mask.masks.cpu().detach().numpy()
        elif isinstance(mask, list):
            mask = np.array(mask)

        # Resize the mask to match the original image size
        if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Convert mask to binary (0 and 1)
        mask = (mask > 0).astype(np.uint8)

        # Apply the mask to the combined image
        combined_image[mask == 1] = image[mask == 1]  # keep the original color where mask is applied

        # Find the center of the mask
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Write the class name to the center of the mask
        cv2.putText(combined_image, f'{class_names[0]}', (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Define the directory to save the images
save_directory = "C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/Snail_segment"

# Save the combined image in the specified directory
cv2.imwrite(f"{save_directory}/combined_image.png", combined_image)

# Load the combined image
combined_image = cv2.imread(f"{save_directory}/combined_image.png")

# Run detection on the combined image with a confidence threshold of 0.89
det_results = det_model.predict(combined_image, conf=0.55)
# print(det_results)

# Store all detected classes
detected_classes = []

# Parse the results
for res in det_results:
    for box in res.boxes.data:
        # Each box is a detection with the following format: [x1, y1, x2, y2, confidence, class]
        x_center, y_center, width, height, conf, class_id = box
        x1 = x_center - (width / 2)
        y1 = y_center - (height / 2)
        x2 = x_center + (width / 2)
        y2 = y_center + (height / 2)
        class_name = det_model.names[int(class_id)]  # Get the class name
        print(f"Detected {class_name} with confidence {conf:.2f}")
        detected_classes.append(class_name)

# Visualize the results
res_plotted = det_results[0].plot()

# Define the save path
save_path = f"{save_directory}/Detect_Segment.jpg"

# Save the image with predictions
cv2.imwrite(save_path, res_plotted)

cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image


# Define two lists to store the labels
seg_labels = []
det_labels = []

# After the segmentations, append the labels to seg_labels list
for result in seg_results:
    for mask in result.masks:
        seg_labels.append(result.names[0])

print("Segmentation results:", seg_labels)

# After the detections, append the labels to det_labels list
for result in det_results:
    for box in result.boxes.data:
        class_name = det_model.names[int(box[5])]  # Get the class name
        det_labels.append(class_name)

print("Detection results:", det_labels)

# Convert the detected_classes list to a string
seg_labels_str = ', '.join(seg_labels)
det_labels_str = ', '.join(det_labels)

print("Segmentation results:", seg_labels_str)
print("Detection results:", det_labels_str)

# Record the start time
start_time = time.time()

# OpenAI
import openai
openai.api_key = "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"

prompt_str = f"You are an agricultural entomologist and also an organic plant producer. Can the species identified as {det_labels_str} be harmful or beneficial to the plants labelled as {seg_labels_str}? If they are harmful, could you explain in an academic manner how to control {det_labels_str} with minimal ecological damage?"

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt_str,
    max_tokens=400,
    temperature=0.1,
    n=1,
)

output = response.choices[0].text.strip()
print(output)

# Visualize
from textwrap import wrap
import numpy as np
import cv2

# Visualize the results
res_plotted = det_results[0].plot()

# Create a black background same size as the prediction image
bg = np.zeros(res_plotted.shape, dtype=np.uint8)

# Wrap the text by a given number of characters
wrapped_text = wrap(output, 50)

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7  # increase the font scale
font_color = (255, 255, 255)
line_type = 2

# Margins for the text
left_margin = int(res_plotted.shape[1] * 0.05)  # 5% of the image width
top_margin = int(res_plotted.shape[0] * 0.05)  # 5% of the image height
y = top_margin  # y position of the first line

for line in wrapped_text:
    # Draw each line on the black background
    cv2.putText(bg, line, (left_margin, y), font, font_scale, font_color, line_type)
    y += 30  # move to the next line

# Combine original image and black background
combined = np.hstack((res_plotted, bg))

# Create a named window where the size can be changed
cv2.namedWindow("Predictions and output", cv2.WINDOW_NORMAL)

# Display the image with predictions and the output text
cv2.imshow("Predictions and output", combined)

# Define the save path
save_path = "C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/Snail_segment/Predictions_and_output.jpg"

# Save the image with predictions and the output text
cv2.imwrite(save_path, combined)

cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image

# Record the end time
end_time = time.time()

# Calculate and print the running time
running_time = end_time - start_time
print(f"The running time of the code is {running_time} seconds.")
