import time
from flask import Flask, json, redirect, render_template, request, Response, jsonify, send_file, send_from_directory, url_for
from flask_cors import CORS
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageDraw, ImageFont
import cv2
import base64
import traceback
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)


frame_width = 640  
frame_height = 480
circle_radius = 300
border_thickness = 10

DATA_DIR = './android'
os.makedirs(DATA_DIR, exist_ok=True)

PICKLE_DIR = os.path.join(os.getcwd(), "data")

cached_user_data = None


if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
#progress = 0 

def generate_video(username):
    global frame_width, frame_height, circle_radius, border_thickness, i, faces_data,text_x,text_y,text_w,text_h
    faces_data = []
    i = 0
    print(frame_width,frame_height)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    progress = 0  
    stop_stream = False

    text = "Insert your face in Circle"

    while cap.isOpened():

        if stop_stream:  
            cv2.rectangle(
                circular_frame,
                (text_x, text_y - text_h),  
                (text_x + text_w, text_y + 10),  
                (0, 0, 0),  
                -1  
            )

            cv2.circle(circular_frame, circle_center, circle_radius, (0,255,0), border_thickness)
            cv2.putText(circular_frame, "Uploading the Frame Wait!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            _, buffer = cv2.imencode('.jpg', circular_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            break 

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        circle_center = (frame_width // 2, frame_height // 2)
        cv2.circle(mask, circle_center, circle_radius, (255, 255, 255), -1)

        if frame.shape[:2] != (frame_height, frame_width):
            frame = cv2.resize(frame, (frame_width, frame_height))

        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        circular_frame = cv2.bitwise_and(frame, frame, mask=mask_gray)

        end_angle_red = int(360 * (100 / 100))
        cv2.ellipse(circular_frame, circle_center, (circle_radius, circle_radius), -90, 0, end_angle_red, (0, 0, 255), border_thickness)
        end_angle_green = int(360 * (progress / 100))
        cv2.ellipse(circular_frame, circle_center, (circle_radius, circle_radius), -90, end_angle_red, end_angle_red + end_angle_green, (0, 255, 0), border_thickness)

        if(len(faces_data) > 0):
            text = "Align your face Properly!"
        

        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w]  
            resized_img = cv2.resize(crop_img, (50, 50))  
            
            if crop_img.shape[0] > 0 or crop_img.shape[1] > 0:
                text = "Hold on Wait! Taking frames"

            if len(faces_data) <= 100 and i % 10 == 0:
                faces_data.append(resized_img)
                text="Hold on Wait! Taking frames"
                #print(text)
                #print(f"if{crop_img}")
                progress = min(100, len(faces_data))

            i += 1  
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.15
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_w,text_h = text_size
        text_x = circle_center[0] + circle_radius - (text_size[0] // 5) + 20   
        text_y = circle_center[1] + circle_radius - 5  

        cv2.putText(circular_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        _, buffer = cv2.imencode('.jpg', circular_frame)
        frame_bytes = buffer.tobytes()


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        

        if len(faces_data) == 100:
            print("❌❌❌Trueee❌❌❌")
            stop_stream = True  
            cv2.rectangle(
                circular_frame,
                (text_x, text_y - text_h),  
                (text_x + text_w, text_y + 10),  
                (0, 0, 0),  
                -1 
            )

            cv2.circle(circular_frame, circle_center, circle_radius, (0,255,0), border_thickness)
            cv2.putText(circular_frame, "Uploading the Frame Wait!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', circular_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            break
                 

    cap.release()
    cv2.destroyAllWindows()

    faces_data_np = np.asarray(faces_data)
    faces_data_np = faces_data_np.reshape(100, -1)

    print(f"faces_data_np: {len(faces_data_np)}, Faces Count: {len(faces_data)}")
    print(faces_data_np)
    if not os.path.exists('data'):
        os.makedirs('data')

    file_name = 'data/names.pkl' 
    #username = 'Ali'
    if not os.path.exists(file_name):
        usernames_list=[username]*100
        with open(file_name, 'wb') as f:
            pickle.dump(usernames_list, f)
    else:
        with open(file_name, 'rb') as f:
            usernames_list = pickle.load(f)
        usernames_list = usernames_list + [username]*100
        with open(file_name, 'wb') as f:
            pickle.dump(usernames_list, f)       

    file_path = 'data/faces_data.pkl'

    if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(faces_data_np, f)
    else:
        try:
            with open(file_path, 'rb') as f:
                faces = pickle.load(f)

            print(f"Existing Faces Shape: {faces.shape}")
            print(f"New Faces Shape: {faces_data_np.shape}")

            if faces.shape[1] == faces_data_np.shape[1]:  
                faces = np.append(faces, faces_data_np, axis=0)
                with open(file_path, 'wb') as f:
                    print("✅ Faces Data Appended Successfully")
                    pickle.dump(faces, f)
            else:
                print("⚠️ Shape Mismatch! Skipping Append.")

        except Exception as e:
            print(f"❌ Error Loading Faces Data: {e}")

            
    yield b"STOP\r\n"
    yield b"" 
    print("✅ Closing connection")
    return
    


def generate_video_FaceRecog():
    global frame_width, frame_height, circle_radius, border_thickness
    
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
    
    print('Shape of Faces matrix --> ', FACES.shape)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    stop_stream = False

    border_color = (0, 0, 255)

    global_text = "Put your face inside that circle"

    global_name = ""
    
    while cap.isOpened():
        if stop_stream:
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (frame_width, frame_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        circle_center = (frame_width // 2, frame_height // 2)
        cv2.circle(mask, circle_center, circle_radius, (255, 255, 255), -1)
        
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        circular_frame = cv2.bitwise_and(frame, frame, mask=mask_gray)

        cv2.circle(circular_frame, circle_center, circle_radius, border_color, border_thickness)

        cv2.putText(circular_frame, global_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)[0]
            
            if output:  
                border_color = (0, 255, 0)
                global_text = "Face detected !"
                global_name = str(output)
                
                

            cv2.putText(circular_frame, global_name, (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
            cv2.circle(circular_frame, circle_center, circle_radius, border_color, border_thickness)

            cv2.putText(circular_frame, str(output), (x - 80, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)
        
        _, buffer = cv2.imencode('.jpg', circular_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    cv2.destroyAllWindows()
    yield b"STOP\r\n"
 

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return 'No frame uploaded', 400

    frame_file = request.files['frame']
    frame = cv2.imdecode(np.frombuffer(frame_file.read(), np.uint8), cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    return Response(frame_bytes, mimetype='image/jpeg')


@app.route('/<namestore>')
def index(namestore):
    print(f"checking the namestore {namestore}")
    return render_template('index.html', namestore=namestore)

@app.route('/main_page')
def main_page():
    print(f"Redirected to main_page with username: ")
    return render_template('main_page.html')

@app.route('/video_feed/<username>',methods=['GET'])
def video_feed(username):
    if not username or username.strip() == "":
        print("❌ ERROR: Username is empty or None!")
        return jsonify({"error": "Username is required"}), 400  

    print(f"🔹 Video feed started for: {username}")

    try:
        return Response(
            generate_video(username),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        print(f"❌ ERROR: Video feed crashed: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/faceRecog')
def faceRecog():
    print(f"checking the detection ")
    return render_template('faceRecog.html')    

@app.route('/video_feed_forRecog',methods=['GET'])
def video_feed_forRecog():
    
    print(f"hitted video thing for FaceRecognistion")
    return Response(generate_video_FaceRecog(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_dimensions', methods=['POST'])
def set_dimensions():
    global frame_width, frame_height
    data = request.json
    print(data)
    frame_width = int(data['width'])
    frame_height = int(data['height'])
    return {'message': 'Dimensions updated successfully'}, 200

@app.route('/send-file/<username>', methods=['POST'])
def receive_face_data(username):
    print("hitted")
    print(username)
    try:

        if request.json is None:
            print("None")
            return jsonify({'error': 'Invalid JSON data received'}), 400
        
        faces_data = request.json.get('faces_data', [])

        if not faces_data:
            print("No face data provided")
            return jsonify({'error': 'No face data provided'}), 400

        print(faces_data)

        print(len(faces_data))   

        faces_data = np.asarray(faces_data)

        
        if faces_data.shape[0] != 100:
            print(f"Invalid number of frames received: {faces_data.shape[0]}")
            return jsonify({'error': 'Expected exactly 100 face frames'}), 400

        
        faces_data = faces_data.reshape(100, -1)

        print(faces_data)

        print(len(faces_data))   

        names_file = os.path.join(DATA_DIR, 'names.pkl')
        if not os.path.exists(names_file):
            names = [username] * 100
            with open(names_file, 'wb') as f:
                pickle.dump(names, f)
        else:
            with open(names_file, 'rb') as f:
                names = pickle.load(f)

            names.extend([username] * 100)

            with open(names_file, 'wb') as f:
                pickle.dump(names, f)

        faces_file = os.path.join(DATA_DIR, 'faces_data.pkl')
        if not os.path.exists(faces_file):
            with open(faces_file, 'wb') as f:
                pickle.dump(faces_data, f)
        else:
            with open(faces_file, 'rb') as f:
                existing_faces = pickle.load(f)

            existing_faces = np.append(existing_faces, faces_data, axis=0)

            with open(faces_file, 'wb') as f:
                pickle.dump(existing_faces, f)

        return jsonify({'message': 'Face data saved successfully'}), 200

    except Exception as e:
        print("Exection occurs")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download/name')
def download_name(): 
    print("Hitted names file")
    try:
        with open("data/names.pkl", "rb") as f:
            names = pickle.load(f)  

        print("Raw names list:", names)

        names_json = json.dumps(names, indent=4)

        print("Converted JSON:\n", names_json)

        return jsonify(names)
    except Exception as e:
        print(f"Error in my name.pkl file {e}")
        return str(e), 404

@app.route('/download/faces_data')
def download_faces_data():
    print("Hitted faces_data file")
    try:
        with open("data/faces_data.pkl", "rb") as f:
            faces_data = pickle.load(f)  

        print("Raw faces_data list:", faces_data)   

        faces_data_list = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in faces_data]

        print("Converted JSON:\n", faces_data_list)

        return jsonify(faces_data_list) 
    except Exception as e:
        print(f"Error in my faces_data.pkl file {e}")
        return str(e), 404

@app.route('/download/new')
def new():
    global cached_user_data
    if cached_user_data is not None:
        return jsonify(cached_user_data)  

    try:
        with open("android/faces_data.pkl", "rb") as f:
            faces_data = pickle.load(f)
        with open("android/names.pkl", "rb") as f:
            names = pickle.load(f)

        faces_dict = {}
        for i, name in enumerate(names):
            faces_dict.setdefault(name, []).append(faces_data[i].tolist())

        cached_user_data = faces_dict 
        return jsonify(faces_dict)
    except Exception as e:
        return jsonify({"error": str(e)}), 500        



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
