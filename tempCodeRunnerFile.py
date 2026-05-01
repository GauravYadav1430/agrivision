import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("[SYSTEM] Booting AgriVision Inference Engine v3.0...")

# Load Models
potato_model = tf.keras.models.load_model('potato_model_best.keras')
gatekeeper_model = MobileNetV2(weights='imagenet')
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Load Facial Recognition AI
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- STAGE 1: FACIAL RECOGNITION (Catches Messi & Magnus) ---
def contains_human_face(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces of varying sizes
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

# --- STAGE 2: SEMANTIC SCANNER (Catches Banana Leaves, Dogs, Jerseys) ---
def is_safe_semantic_context(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    preds = gatekeeper_model.predict(x, verbose=0)
    
    # Get the top 5 predictions
    top_5 = decode_predictions(preds, top=5)[0]
    
    # Define absolute dealbreakers
    blacklisted_words = [
        'banana', 'jersey', 'suit', 'shirt', 'ball', 'racket', 'sunglasses', 
        'wig', 'microphone', 'desk', 'laptop', 'car', 'dog', 'cat', 'person'
    ]
    
    # If the #1 prediction is an animal (ImageNet classes 0-397)
    if np.argmax(preds[0]) <= 397:
        return False, f"Animal detected ({top_5[0][1].replace('_', ' ')})"

    # Scan the top 5 predictions for blacklisted items
    for _, label, confidence in top_5:
        if confidence > 0.05: # If it's even 5% sure it sees a bad item
            if any(bad_word in label.lower() for bad_word in blacklisted_words):
                return False, f"Non-potato object detected ({label.replace('_', ' ')})"
                
    return True, "Context Safe"

# --- STAGE 3: ORGANIC COLOR FILTER ---
def contains_organic_matter(filepath):
    cv_img = cv2.imread(filepath)
    hsv = cv2.cvtColor(cv2.resize(cv_img, (256, 256)), cv2.COLOR_BGR2HSV)
    
    # Broad organic range (Greens, Yellows, Browns)
    lower_bound = np.array([10, 20, 20])
    upper_bound = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    ratio = (cv2.countNonZero(mask) / (256*256)) * 100
    return ratio > 10.0 # Must be at least 10% organic colors

# --- SEVERITY CALCULATOR ---
def calculate_severity(image_path):
    img = cv2.imread(image_path)
    if img is None: return 0.0
    
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    leaf_mask = cv2.inRange(hsv, np.array([10, 20, 20]), np.array([100, 255, 255]))
    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    if total_leaf_pixels == 0: return 0.0
        
    disease_mask = cv2.inRange(hsv, np.array([10, 50, 20]), np.array([30, 255, 200]))
    actual_disease = cv2.bitwise_and(disease_mask, disease_mask, mask=leaf_mask)
    diseased_pixels = cv2.countNonZero(actual_disease)
    
    return float(round((diseased_pixels / total_leaf_pixels) * 100, 2))

# --- ROUTES ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file selected"}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    # 1. Face Detection (Magnus & Messi Blocker)
    if contains_human_face(filepath):
        return jsonify({
            "disease": "Invalid Input", "confidence": 0, "severity": 0,
            "status": "Human face detected. Please upload a clear image of a leaf."
        })

    # 2. Semantic Scanner (Banana Leaf & Random Object Blocker)
    is_safe, context_msg = is_safe_semantic_context(filepath)
    if not is_safe:
        return jsonify({
            "disease": "Invalid Input", "confidence": 0, "severity": 0,
            "status": f"Rejected: {context_msg}."
        })

    # 3. Organic Filter
    if not contains_organic_matter(filepath):
        return jsonify({
            "disease": "Invalid Input", "confidence": 0, "severity": 0,
            "status": "Insufficient plant matter detected."
        })

    # 4. Disease Classification
    img = image.load_img(filepath, target_size=(256, 256))
    img_array = tf.expand_dims(image.img_to_array(img), 0)
    
    predictions = potato_model.predict(img_array, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(round(100 * np.max(predictions[0]).item(), 2))

    # Strict Confidence Check
    if confidence < 75.0:
        return jsonify({
            "disease": "Uncertain Diagnosis", "confidence": confidence, "severity": 0,
            "status": "Model entropy too high. This may be an unknown plant species or a blurry image."
        })

    # 5. Severity & Output
    if "healthy" in predicted_class.lower():
        severity = 0.0
        status = "Plant is healthy! Maintain current conditions."
    else:
        severity = calculate_severity(filepath)
        if severity < 10.0: status = "Mild infection. Monitor closely."
        elif severity < 30.0: status = "Moderate infection. Treat soon."
        else: status = "Severe infection. Immediate action required."

    display_class = predicted_class.replace('Potato___', '').replace('_', ' ').title()

    return jsonify({
        "disease": display_class, "confidence": confidence, "severity": severity, "status": status
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)