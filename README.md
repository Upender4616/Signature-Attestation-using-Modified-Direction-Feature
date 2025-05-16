# Signature Attestation System
A Flask web application to verify the similarity between two signature images using computer vision.
#Requirements
1.python 3.9 or latest version
2.Visual Studio Code or Command Prompt
## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
## Features
- Upload and compare two signatures
- ORB feature matching and similarity scoring
- Stroke-based velocity and pressure analysis
- Displays visual results of comparison
## Installation
1. Clone the repo or download the ZIP.
2. Navigate to the project folder.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
#Run the app
python app.py
#Project Structure
signature_attestation_project/
│
├── app.py # Main Flask application
├── requirements.txt # Required Python packages
├── README.md # Project overview and instructions
└── templates/ # HTML templates for rendering views
├── upload.html
└── result.html

