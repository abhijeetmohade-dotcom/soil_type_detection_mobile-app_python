from flask import Flask, request, render_template, jsonify
import os
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='templates', static_folder='static')

# ---- STEP 1: Mapping Data ----
fertilizer_recommendations = {
    "Black Soil": ["NPK 20-20-0", "Urea", "Super phosphate", "Potash"],
    "Red Soil": ["Complex NPK", "DAP", "Potassium sulfate"],
    "Laterite Soil": ["NPK 12-32-16", "Rock phosphate", "Lime"],
    "Alluvial Soil": ["Urea", "Super phosphate", "Potash"],
    "Peat Soil": ["Urea", "Lime", "Phosphate"],
    "Cinder Soil": ["Balanced NPK", "Potash"],
    "Yellow Soil": ["NPK (12-32-16)", "Magnesium Sulfate"]
}

soil_content_ideals = {
    "Black Soil": {
        "pH": "6.5 – 7.5",
        "Organic Carbon": "0.75 – 1%",
        "Nitrogen": "300 – 450 kg/ha",
        "Phosphorus": "20 – 25 kg/ha",
        "Potassium": "250 – 350 kg/ha"
    },
    "Red Soil": {
        "pH": "5.5 – 7.0",
        "Organic Carbon": "0.5 – 0.8%",
        "Nitrogen": "250 – 400 kg/ha",
        "Phosphorus": "15 – 20 kg/ha",
        "Potassium": "150 – 200 kg/ha"
    },
    "Laterite Soil": {
        "pH": "5.0 – 6.5",
        "Organic Carbon": "0.5 – 1%",
        "Nitrogen": "200–350 kg/ha",
        "Phosphorus": "10–18 kg/ha",
        "Potassium": "120–180 kg/ha"
    },
    "Alluvial Soil": {
        "pH": "6.8 – 8.0",
        "Organic Carbon": "0.5 – 0.7%",
        "Nitrogen": "220 – 380 kg/ha",
        "Phosphorus": "16 – 22 kg/ha",
        "Potassium": "180 – 280 kg/ha"
    },
    "Peat Soil": {
        "pH": "5.0 – 6.5",
        "Organic Carbon": "2 – 5%",
        "Nitrogen": "200 – 320 kg/ha",
        "Phosphorus": "12 – 18 kg/ha",
        "Potassium": "110 – 200 kg/ha"
    },
    "Cinder Soil": {
        "pH": "6.0 – 7.4",
        "Organic Carbon": "1 – 2%",
        "Nitrogen": "150 – 250 kg/ha",
        "Phosphorus": "10 – 18 kg/ha",
        "Potassium": "80 – 140 kg/ha"
    },
    "Yellow Soil": {
        "pH": "5.0 – 6.8",
        "Organic Carbon": "0.5 – 0.8%",
        "Nitrogen": "140 – 260 kg/ha",
        "Phosphorus": "10 – 16 kg/ha",
        "Potassium": "100 – 180 kg/ha"
    }
}

def preprocess(img_path, target_size=(224,224)):
    img = Image.open(img_path).convert('RGB').resize(target_size)
    arr = np.array(img) / 255.0
    return arr.flatten().reshape(1, -1)

def predict_soil(test_img_path, data_dir='soil-data', target_size=(224,224)):
    best_score = -1
    best_class = None
    test_vec = preprocess(test_img_path, target_size)
    for soil_class in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, soil_class)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                train_vec = preprocess(img_path, target_size)
                score = cosine_similarity(test_vec, train_vec)[0][0]
                if score > best_score:
                    best_score = score
                    best_class = soil_class
    return best_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        f = request.files['image']
        filename = f.filename
        upload_path = os.path.join('uploads', filename)
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        f.save(upload_path)
        soil_class = predict_soil(upload_path)
        fertilizers = fertilizer_recommendations.get(soil_class, [])
        soil_ideals = soil_content_ideals.get(soil_class, {})
        return jsonify({
            "predicted_class": soil_class,
            "recommended_fertilizers": fertilizers,
            "soil_contents": soil_ideals
        })
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

