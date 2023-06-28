# Image Enhancement Web App

This is a web application built with Flask that allows users to enhance their images using different models. The app supports various image enhancement models, including RealESRGAN and GFPGAN.
Prerequisites

Before running the web app, make sure you have the following dependencies installed:

You can install the dependencies by running the following command:

```pip install -r requirements.txt```

## Usage

Clone this repository to your local machine:

## bash

```git clone https://github.com/hakunamatata1997/Image-Enhancer-WebApp.git```

Navigate to the project directory:

```cd image-enhancement-webapp```

Run the Flask app:

```python app.py```

Open your web browser and go to http://localhost:52525.

Upload an image file (.jpg, .jpeg, or .png) and select the desired model version and scaling factor.

Click the "Enhance" button to process the image and view the enhanced result.

Folder Structure:

app.py: The main Flask application file.
templates/: Directory containing the HTML templates for the web pages.
static/: Directory containing the CSS stylesheets and images used in the web pages.
uploads/: Directory where user-uploaded images are stored.
outputs/: Directory where the enhanced images are saved.

Customization:

You can customize the supported models by adding or modifying the models in the enhance_image function in app.py.
The web app's appearance and layout can be modified by editing the HTML templates in the templates/ directory and the CSS stylesheets in the static/ directory.
