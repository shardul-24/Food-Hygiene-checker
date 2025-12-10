# import os
# import base64
# import re
# from datetime import datetime
# from flask import Flask, render_template, request, redirect, url_for, session, flash
# from werkzeug.utils import secure_filename
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np

# # ------------------------
# # Firebase Setup
# # ------------------------
# import firebase_admin
# from firebase_admin import credentials, firestore

# cred = credentials.Certificate("firebase_key.json")  # your Firebase key file
# firebase_admin.initialize_app(cred)
# db = firestore.client()

# # ------------------------
# # Flask Setup
# # ------------------------
# app = Flask(__name__)
# app.secret_key = "your_secret_key"
# UPLOAD_FOLDER = "static/uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# # ------------------------
# # Load Model
# # ------------------------
# model_path = "best_hygiene_model.keras"
# if os.path.exists(model_path):
#     model = load_model(model_path)
# else:
#     model = None

# class_labels = ["Good", "Moderate", "Poor"]

# # ------------------------
# # Routes
# # ------------------------
# @app.route("/", methods=["GET", "POST"])
# def index():
#     # Ensure user is logged in
#     if "user_id" not in session:
#         return redirect(url_for("login"))

#     prediction, confidence, uploaded_image, questionnaire, probs = None, None, None, {}, None

#     if request.method == "POST":
#         file = request.files.get("file")
#         webcam_image = request.form.get("webcam_image")

#         # Save uploaded image
#         uploaded_image_path = None
#         if file and file.filename != "":
#             filename = secure_filename(file.filename)
#             uploaded_image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#             file.save(uploaded_image_path)
#         elif webcam_image and webcam_image != "":
#             # Handle webcam capture safely
#             try:
#                 if "," in webcam_image:
#                     _, encoded = webcam_image.split(",", 1)
#                 else:
#                     encoded = webcam_image
#                 filename = f"webcam_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
#                 uploaded_image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#                 with open(uploaded_image_path, "wb") as f:
#                     f.write(base64.b64decode(encoded))
#             except Exception as e:
#                 flash(f"❌ Webcam image capture failed: {str(e)}")

#         # Questionnaire responses
#         questionnaire = {
#             "shop_name": request.form.get("shop_name"),
#             "shop_address": request.form.get("shop_address"),
#             "city": request.form.get("city"),
#             "surroundings": request.form.get("surroundings"),
#             "utensils": request.form.get("utensils"),
#             "covered_food": request.form.get("covered_food"),
#             "others_hygiene": request.form.get("others_hygiene"),
#         }

#         # Image Prediction
#         image_score, probs = 0, {}
#         if uploaded_image_path and model:
#             img = load_img(uploaded_image_path, target_size=(128, 128))
#             img_array = img_to_array(img) / 255.0
#             img_array = np.expand_dims(img_array, axis=0)

#             predictions = model.predict(img_array)[0]
#             class_idx = np.argmax(predictions)
#             prediction = class_labels[class_idx]
#             image_score = float(predictions[class_idx]) * 100
#             confidence = round(image_score, 2)

#             probs = {class_labels[i]: round(float(predictions[i]) * 100, 2) for i in range(len(class_labels))}

#         # Questionnaire Scoring
#         score_map = {"Good": 100, "Moderate": 50, "Poor": 20, "Yes": 100, "No": 30}
#         question_score = sum(score_map.get(ans, 50) for ans in [
#             questionnaire["surroundings"],
#             questionnaire["utensils"],
#             questionnaire["covered_food"],
#             questionnaire["others_hygiene"]
#         ]) / 4

#         # Final blended score
#         final_score = ((image_score * 0.5) + (question_score * 0.5)) if image_score > 0 else question_score

#         # Final Hygiene Class
#         if final_score >= 70:
#             prediction = "Good"
#         elif final_score >= 40:
#             prediction = "Moderate"
#         else:
#             prediction = "Poor"

#         confidence = round(final_score, 2)

#         # Save result to Firebase
#         db.collection("results").add({
#             "user_id": session["user_id"],
#             "hygiene_class": prediction,
#             "score": final_score,
#             "shop_name": questionnaire["shop_name"],
#             "shop_address": questionnaire["shop_address"],
#             "city": questionnaire["city"],
#             "surroundings": questionnaire["surroundings"],
#             "utensils": questionnaire["utensils"],
#             "covered_food": questionnaire["covered_food"],
#             "others_hygiene": questionnaire["others_hygiene"],
#             "uploaded_image": uploaded_image_path,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         })

#         uploaded_image = uploaded_image_path

#     return render_template("index.html", prediction=prediction, confidence=confidence,
#                            uploaded_image=uploaded_image, questionnaire=questionnaire, probs=probs)

# # ------------------------
# # Dashboard
# # ------------------------
# @app.route("/dashboard")
# def dashboard():
#     if "user_id" not in session:
#         return redirect(url_for("login"))

#     good_count = len(db.collection("results").where("hygiene_class", "==", "Good").get())
#     moderate_count = len(db.collection("results").where("hygiene_class", "==", "Moderate").get())
#     poor_count = len(db.collection("results").where("hygiene_class", "==", "Poor").get())

#     reports_ref = db.collection("results").order_by("timestamp", direction=firestore.Query.DESCENDING).get()
#     reports = [doc.to_dict() for doc in reports_ref]

#     return render_template("dashboard.html",
#                            good_count=good_count,
#                            moderate_count=moderate_count,
#                            poor_count=poor_count,
#                            reports=reports)

# # ------------------------
# # User Auth
# # ------------------------
# @app.route("/login", methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         username = request.form.get("username")
#         password = request.form.get("password")

#         users_ref = db.collection("users")
#         query = users_ref.where("username", "==", username).where("password", "==", password).get()

#         if query:
#             session["user_id"] = query[0].id
#             return redirect(url_for("index"))
#         else:
#             flash("❌ Invalid username or password.")
#             return redirect(url_for("login"))
#     return render_template("login.html")


# @app.route("/register", methods=["GET", "POST"])
# def register():
#     # Prevent access if user already logged in
#     if "user_id" in session:
#         return redirect(url_for("index"))

#     if request.method == "POST":
#         username = request.form.get("username")
#         password = request.form.get("password")

#         if not re.match(r'^(?=.*[!@#$%^&*(),.?":{}|<>]).{6,12}$', password):
#             flash("❌ Password must be 6–12 characters long and include at least one special character.")
#             return redirect(url_for("register"))

#         users_ref = db.collection("users")
#         existing_user = users_ref.where("username", "==", username).get()
#         if existing_user:
#             flash("⚠️ Username already exists, try another.")
#             return redirect(url_for("register"))

#         users_ref.add({"username": username, "password": password})
#         flash("✅ Registration successful! Please login.")
#         return redirect(url_for("login"))

#     return render_template("register.html")


# @app.route("/logout")
# def logout():
#     session.clear()
#     return redirect(url_for("login"))

# # ------------------------
# if __name__ == "__main__":
#     app.run(debug=True)


import os
import base64
import re
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# ------------------------
# Firebase Setup
# ------------------------
import firebase_admin
from firebase_admin import credentials, firestore

# Use environment variable path if deployed
firebase_key_path = os.environ.get("FIREBASE_KEY_PATH", "firebase_key.json")

cred = credentials.Certificate(firebase_key_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ------------------------
# Flask Setup
# ------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------------
# Load Model
# ------------------------
model_path = "best_hygiene_model.keras"

model = None
if os.path.exists(model_path):
    print("📌 Loading TensorFlow model...")
    model = load_model(model_path)
    print("✔ Model Loaded Successfully!")
else:
    print("❌ Model file not found!")

class_labels = ["Good", "Moderate", "Poor"]

# ------------------------
# Routes
# ------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))

    prediction, confidence, uploaded_image, questionnaire, probs = None, None, None, {}, None

    if request.method == "POST":
        file = request.files.get("file")
        webcam_image = request.form.get("webcam_image")

        uploaded_image_path = None

        # Uploaded image file
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            uploaded_image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(uploaded_image_path)

        # Webcam image
        elif webcam_image and webcam_image != "":
            try:
                if "," in webcam_image:
                    _, encoded = webcam_image.split(",", 1)
                else:
                    encoded = webcam_image

                filename = f"webcam_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                uploaded_image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                with open(uploaded_image_path, "wb") as f:
                    f.write(base64.b64decode(encoded))
            except Exception as e:
                flash(f"❌ Webcam image capture failed: {str(e)}")

        # Questionnaire
        questionnaire = {
            "shop_name": request.form.get("shop_name"),
            "shop_address": request.form.get("shop_address"),
            "city": request.form.get("city"),
            "surroundings": request.form.get("surroundings"),
            "utensils": request.form.get("utensils"),
            "covered_food": request.form.get("covered_food"),
            "others_hygiene": request.form.get("others_hygiene"),
        }

        # Image Prediction
        image_score = 0
        probs = {}

        if uploaded_image_path and model:
            img = load_img(uploaded_image_path, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)[0]
            class_idx = np.argmax(predictions)
            prediction = class_labels[class_idx]
            image_score = float(predictions[class_idx]) * 100
            confidence = round(image_score, 2)

            probs = {class_labels[i]: round(float(predictions[i]) * 100, 2)
                     for i in range(len(class_labels))}

        # Questionnaire Score
        score_map = {"Good": 100, "Moderate": 50, "Poor": 20, "Yes": 100, "No": 30}
        question_score = sum(score_map.get(ans, 50) for ans in [
            questionnaire["surroundings"],
            questionnaire["utensils"],
            questionnaire["covered_food"],
            questionnaire["others_hygiene"],
        ]) / 4

        final_score = ((image_score * 0.5) + (question_score * 0.5)) if image_score > 0 else question_score

        if final_score >= 70:
            prediction = "Good"
        elif final_score >= 40:
            prediction = "Moderate"
            confidence = round(final_score, 2)
        else:
            prediction = "Poor"

        # Save in Firebase
        db.collection("results").add({
            "user_id": session["user_id"],
            "hygiene_class": prediction,
            "score": final_score,
            "shop_name": questionnaire["shop_name"],
            "shop_address": questionnaire["shop_address"],
            "city": questionnaire["city"],
            "surroundings": questionnaire["surroundings"],
            "utensils": questionnaire["utensils"],
            "covered_food": questionnaire["covered_food"],
            "others_hygiene": questionnaire["others_hygiene"],
            "uploaded_image": uploaded_image_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        uploaded_image = uploaded_image_path

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           uploaded_image=uploaded_image,
                           questionnaire=questionnaire,
                           probs=probs)

# ------------------------
# Dashboard
# ------------------------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    good_count = len(db.collection("results").where("hygiene_class", "==", "Good").get())
    moderate_count = len(db.collection("results").where("hygiene_class", "==", "Moderate").get())
    poor_count = len(db.collection("results").where("hygiene_class", "==", "Poor").get())

    reports_ref = db.collection("results").order_by("timestamp", direction=firestore.Query.DESCENDING).get()
    reports = [doc.to_dict() for doc in reports_ref]

    return render_template("dashboard.html",
                           good_count=good_count,
                           moderate_count=moderate_count,
                           poor_count=poor_count,
                           reports=reports)

# ------------------------
# User Authentication
# ------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        users_ref = db.collection("users")
        query = users_ref.where("username", "==", username).where("password", "==", password).get()

        if query:
            session["user_id"] = query[0].id
            return redirect(url_for("index"))
        else:
            flash("❌ Invalid username or password.")
            return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # password rule
        if not re.match(r'^(?=.*[!@#$%^&*(),.?":{}|<>]).{6,12}$', password):
            flash("❌ Password must be 6–12 characters long and include at least one special character.")
            return redirect(url_for("register"))

        users_ref = db.collection("users")
        existing_user = users_ref.where("username", "==", username).get()

        if existing_user:
            flash("⚠️ Username already exists.")
            return redirect(url_for("register"))

        users_ref.add({"username": username, "password": password})
        flash("✅ Registration successful! Please login.")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ------------------------
# Deployment Entry Point
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)












