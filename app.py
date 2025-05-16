from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration for file uploads and static files
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create necessary folders if they don't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Image preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (300, 150))
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary_image

# Feature extraction using ORB and match calculation
def extract_and_match_features(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

# Function to display matched images with the similarity result
def display_images_with_matches(img1, img2, kp1, kp2, matches, match_result):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    text_position = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0) if match_result == "Match" else (0, 0, 255)
    thickness = 2
    line_type = cv2.LINE_AA
    cv2.putText(img_matches, match_result, text_position, font, font_scale, font_color, thickness, line_type)
    result_path = os.path.join(app.config['STATIC_FOLDER'], 'result.png')
    cv2.imwrite(result_path, img_matches)
    return result_path

# Compare signatures and return similarity score and result image
def compare_signatures(image_path1, image_path2, threshold=100):
    img1 = preprocess_image(image_path1)
    img2 = preprocess_image(image_path2)
    kp1, kp2, matches = extract_and_match_features(img1, img2)
    similarity_score = len(matches)
    match_result = "Match" if similarity_score > threshold else "Do Not Match"
    result_path = display_images_with_matches(img1, img2, kp1, kp2, matches, match_result)
    return similarity_score, result_path

# Extract strokes and contours from the signature image
def extract_strokes(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Estimate velocity of the strokes based on contour distance
def estimate_velocity(contours):
    velocities = []
    for contour in contours:
        distances = [cv2.norm(contour[i] - contour[i-1]) for i in range(1, len(contour))]
        velocities.extend(distances)
    average_velocity = np.mean(velocities) if velocities else 0
    return average_velocity

# Estimate pressure of the strokes based on bounding box width
def estimate_pressure(contours):
    pressures = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        pressures.append(w)
    average_pressure = np.mean(pressures) if pressures else 0
    return average_pressure

# Function to save image with strokes/contours
def display_strokes(image, contours, filename):
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_with_contours = cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
    result_path = os.path.join(app.config['STATIC_FOLDER'], f"{filename}.png")
    cv2.imwrite(result_path, image_with_contours)
    return result_path

# Analyze a single signature and return velocity, pressure, and image with strokes
def analyze_signature(image_path, filename):
    binary_image = preprocess_image(image_path)
    contours = extract_strokes(binary_image)
    velocity = estimate_velocity(contours)
    pressure = estimate_pressure(contours)
    result_path = display_strokes(binary_image, contours, filename)
    return velocity, pressure, result_path

# Route to handle file uploads
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')

        # Check if files were uploaded
        if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
            # Secure file names and save them
            file1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file1.filename))
            file2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file2.filename))
            file1.save(file1_path)
            file2.save(file2_path)

            # Process the images and compare them
            score, match_result_path = compare_signatures(file1_path, file2_path)
            velocity1, pressure1, strokes1_path = analyze_signature(file1_path, "stroke1")
            velocity2, pressure2, strokes2_path = analyze_signature(file2_path, "stroke2")

            return render_template('result.html',
                                   score=score,
                                   match_result_path=match_result_path,
                                   velocity1=velocity1, pressure1=pressure1, strokes1_path=strokes1_path,
                                   velocity2=velocity2, pressure2=pressure2, strokes2_path=strokes2_path)
        else:
            return "Invalid file format. Please upload valid image files."

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
