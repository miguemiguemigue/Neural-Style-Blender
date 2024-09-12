# Neural Style Blender

This project implements Neural Style Transfer (NST) using a custom model based on VGG. It allows users to upload content and style images via a web interface and apply the style of one image onto another.

Project Structure:
- model_training/: Contains the script to train the model and save the weights.
- flask_app/: The Flask application that provides an API for performing inference.
- web_frontend/: Web application for user interaction (can also be part of Flask).
- utils/: Utility functions for image preprocessing and model loading.
