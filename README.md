
# AgroGPT

AgroGPT is a comprehensive image analysis and machine learning automation tool designed to aid agricultural entomologists and organic plant producers. Utilizing state-of-the-art object detection and segmentation models, AgroGPT processes imagery to identify and classify species that may impact plant health.

## Features

- **Image Segmentation**: Using custom-trained YOLO models for precise segmentation of agricultural images.
- **Object Detection**: Detect various species within an image with high confidence using YOLO models.
- **Analysis Automation**: Integration with OpenAI's GPT-3 to automatically analyze and determine the impact of detected species on plant health.
- **Result Visualization**: Visual representation of the analysis alongside the original image for easier interpretation.

## Prerequisites

Before running AgroGPT, ensure that the following libraries are installed:
- OpenCV
- PyTorch
- Ultralytics YOLO
- Numpy

You can install these with the following command:

```sh
pip install opencv-python torch numpy ultralytics
```

**Note**: This project is released under the GNU General Public License (GPL), which ensures freedom to share and change all versions of a program--to make sure it remains free software for all its users.

## Usage

To use AgroGPT, provide the paths to your trained model weights and the image you wish to process. The script will perform segmentation and detection, then query OpenAI's GPT-3 with the results to provide an expert analysis.

## Installation

Clone the repository to your local machine:

```sh
git clone https://github.com/yavuzxyz/agrogpt.git
```

Navigate to the cloned directory, and install the required packages:

```sh
pip install -r requirements.txt
```

## Running the Script

Execute the script with Python:

```sh
python AgroGPT.py
```

## Output

The script will output the processed image with detected objects and a textual analysis of the results. It also saves the output images to the specified directory.

## License

This project is licensed under the GNU GPL v3.0. For more details, see [LICENSE](LICENSE).

## Contact

For support or to contribute to the project, please reach out to the repository owner at yavuz@example.com.
