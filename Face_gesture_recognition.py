import tkinter as tk
from tkinter import messagebox
import cv2
from deepface import DeepFace
import json
from PIL import Image, ImageTk  # Importing for handling background image

# File to store the registered gesture
registered_emotion_file = "registered_emotion.json"

# Function to save the registered emotion
def save_registered_emotion(emotion):
    with open(registered_emotion_file, 'w') as file:
        json.dump({"registered_emotion": emotion}, file)

# Function to load the registered emotion
def load_registered_emotion():
    try:
        with open(registered_emotion_file, 'r') as file:
            data = json.load(file)
            return data.get("registered_emotion", None)
    except FileNotFoundError:
        return None

# Function for capturing emotions
def capture_emotion(mode):
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    detected_emotion = None

    while True:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            detected_emotion = result[0]['dominant_emotion']

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Emotion Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit capture
            break

    cap.release()
    cv2.destroyAllWindows()

    if mode == 'register':
        if detected_emotion:
            save_registered_emotion(detected_emotion)
            messagebox.showinfo("Registration", f"Emotion '{detected_emotion}' registered successfully!")
        else:
            messagebox.showerror("Error", "No emotion detected. Please try again.")
    elif mode == 'login':
        registered_emotion = load_registered_emotion()
        if not registered_emotion:
            messagebox.showerror("Error", "No registered emotion found. Please register first.")
        elif detected_emotion == registered_emotion:
            messagebox.showinfo("Login", "Login successful! Welcome!")
            show_welcome_page()
        else:
            messagebox.showerror("Error", "Emotion does not match the registered gesture.")

# Welcome page
def show_welcome_page():
    for widget in root.winfo_children():
        widget.destroy()
    tk.Label(root, text="Welcome to the Emotion-Based Login System!", font=("Arial", 18, "bold"), fg="white", bg="black").pack(pady=20)
    tk.Button(root, text="Exit", command=root.destroy, font=("Arial", 14, "bold"), fg="white", bg="red").pack(pady=20)

# Main application
root = tk.Tk()
root.title("Emotion-Based Login System")
root.geometry("500x400")

# Set background color and image
root.configure(bg='black')

# Load and set the robot background image
bg_image = Image.open("robot.jpg")  # Provide path to your image file
bg_image = bg_image.resize((500, 400), Image.LANCZOS)  # Resize the image to fit the window size
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a label to display the background image
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)  # Make the background cover the whole window

# Create a frame for the buttons to appear on top of the background
frame = tk.Frame(root, bg='black')
frame.place(relwidth=1, relheight=1)

# Title label
tk.Label(frame, text="Emotion-Based Login System", font=("Arial", 18, "bold"), fg="white", bg="black").pack(pady=20)

# Buttons with styling
tk.Button(frame, text="Register Emotion Gesture", command=lambda: capture_emotion('register'), font=("Arial", 14, "bold"), fg="black", bg="yellow").pack(pady=10)
tk.Button(frame, text="Login with Emotion Gesture", command=lambda: capture_emotion('login'), font=("Arial", 14, "bold"), fg="black", bg="yellow").pack(pady=10)
tk.Button(frame, text="Exit", command=root.destroy, font=("Arial", 14, "bold"), fg="black", bg="red").pack(pady=20)

root.mainloop()
