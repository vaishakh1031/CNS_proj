import cv2
import mediapipe as mp
import json
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Directory to save user facial gestures
REGISTERED_GESTURES_DIR = "registered_gestures"
GESTURES_JSON_FILE = "gestures.json"

# Create the directory if it doesn't exist
if not os.path.exists(REGISTERED_GESTURES_DIR):
    os.makedirs(REGISTERED_GESTURES_DIR)

# Function to save gesture data to JSON
def save_gesture_to_json(gesture_name, gesture_path):
    # Load existing gestures from the file
    if os.path.exists(GESTURES_JSON_FILE):
        with open(GESTURES_JSON_FILE, "r") as f:
            gestures_data = json.load(f)
    else:
        gestures_data = {}

    # Add the new gesture data
    gestures_data[gesture_name] = gesture_path

    # Save back to the JSON file
    with open(GESTURES_JSON_FILE, "w") as f:
        json.dump(gestures_data, f, indent=4)

# Function to load registered gestures from JSON
def load_registered_gestures():
    if os.path.exists(GESTURES_JSON_FILE):
        with open(GESTURES_JSON_FILE, "r") as f:
            return json.load(f)
    return {}

class FaceGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Gesture Registration and Login")
        self.root.geometry("400x300")
        
        # Mediapipe initialization
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)
        
        # Labels and buttons
        self.label = tk.Label(root, text="Face Gesture Recognition", font=("Helvetica", 16))
        self.label.pack(pady=20)
        
        self.register_button = tk.Button(root, text="Register Gesture", command=self.register_gesture, font=("Helvetica", 14))
        self.register_button.pack(pady=10)
        
        self.login_button = tk.Button(root, text="Login with Gesture", command=self.login_with_gesture, font=("Helvetica", 14))
        self.login_button.pack(pady=10)
    
    def capture_face_gesture(self, gesture_name):
        """
        Captures an image of the user's face for a given gesture.
        """
        cap = cv2.VideoCapture(0)
        cv2.namedWindow(f"Capture Gesture - {gesture_name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for Mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)
            
            # Draw face detection box
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                           int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, bbox, (0, 255, 0), 2)
            
            cv2.imshow(f"Capture Gesture - {gesture_name}", frame)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):  # Press 's' to save the gesture image
                gesture_path = os.path.join(REGISTERED_GESTURES_DIR, f"{gesture_name}.jpg")
                cv2.imwrite(gesture_path, frame)
                # Save the gesture data to JSON
                save_gesture_to_json(gesture_name, gesture_path)
                messagebox.showinfo("Success", f"Gesture '{gesture_name}' registered successfully!")
                break
            elif key & 0xFF == ord('q'):  # Press 'q' to quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def register_gesture(self):
        """
        Handles gesture registration.
        """
        gesture_name = tk.simpledialog.askstring("Register Gesture", "Enter Gesture Name (e.g., smile, anger):")
        if gesture_name:
            self.capture_face_gesture(gesture_name)
    
    def login_with_gesture(self):
        """
        Handles gesture login.
        """
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Login - Capture Gesture")
        
        matched_gesture = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for Mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(frame_rgb)
            
            # Draw face detection box
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                           int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, bbox, (0, 255, 0), 2)
            
            cv2.imshow("Login - Capture Gesture", frame)
            
            # Compare frame with registered gestures
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):  # Press 's' to check login
                matched_gesture = self.compare_gestures(frame)
                if matched_gesture:
                    self.display_welcome_page(matched_gesture)
                else:
                    messagebox.showerror("Login Failed", "No matching gesture found!")
                break
            elif key & 0xFF == ord('q'):  # Press 'q' to quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def compare_gestures(self, current_frame):
        """
        Compares the current frame with registered gestures.
        Returns the name of the matched gesture or None.
        """
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        best_match_score = 0
        matched_gesture = None

        # Load all registered gestures from JSON
        gestures_data = load_registered_gestures()

        for gesture_name, gesture_path in gestures_data.items():
            # Check if the gesture image exists
            if not os.path.exists(gesture_path):
                print(f"Warning: {gesture_path} does not exist.")
                continue

            gesture_image = cv2.imread(gesture_path, cv2.IMREAD_GRAYSCALE)
            
            # Ensure the image is loaded properly
            if gesture_image is None:
                print(f"Error: Unable to load image {gesture_path}")
                continue

            # Resize images to match dimensions
            gesture_image_resized = cv2.resize(gesture_image, (gray_current.shape[1], gray_current.shape[0]))
            score, _ = ssim(gray_current, gesture_image_resized, full=True)
            print(f"Gesture: {gesture_name}, SSIM Score: {score}")
            
            if score > best_match_score:
                best_match_score = score
                matched_gesture = gesture_name

        print(f"Best match score: {best_match_score}")
        # Threshold for gesture matching
        if best_match_score > 0.5:  # Adjust this threshold
            return matched_gesture
        return None

    def display_welcome_page(self, matched_gesture):
        """
        Displays a welcome page after successful login.
        """
        welcome_window = tk.Toplevel(self.root)
        welcome_window.title("Welcome")
        welcome_window.geometry("400x300")
        
        welcome_label = tk.Label(welcome_window, text="Welcome to Secure Application", font=("Helvetica", 18))
        welcome_label.pack(pady=20)
        
        gesture_label = tk.Label(welcome_window, text=f"Logged in with gesture: {matched_gesture}", font=("Helvetica", 14))
        gesture_label.pack(pady=10)
        
        exit_button = tk.Button(welcome_window, text="Exit", command=welcome_window.destroy, font=("Helvetica", 14))
        exit_button.pack(pady=20)


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceGestureApp(root)
    root.mainloop()
