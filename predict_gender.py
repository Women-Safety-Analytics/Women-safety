import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import time

# File paths for models and weights
GENDER_MODEL = 'weights/deploy_gender.prototxt'
GENDER_PROTO = 'weights/gender_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Email configuration
SENDER_EMAIL = "vangarajesh181817@gmail.com"
RECEIVER_EMAIL = "leelamanjunathchimirela@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
PASSWORD = "jhui lums xrcw tqmk"#r app-specific password)

# Load face detection and gender models
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def send_email_with_attachment(image_path):
    """Send an email with the captured image attached."""
    try:
        # Setup email parameters
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = "Gender Count Alert - More Females Detected"

        # Attach image
        part = MIMEBase('application', 'octet-stream')
        with open(image_path, 'rb') as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(image_path)}")
        msg.attach(part)

        # Connect to SMTP server and send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

def get_faces(frame, confidence_threshold=0.5):
    """Detect faces in a frame."""
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            start_x, start_y, end_x, end_y = start_x-10, start_y-10, end_x+10, end_y+10
            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(frame.shape[1], end_x)
            end_y = min(frame.shape[0], end_y)
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def get_optimal_font_scale(text, width):
    """Determine the optimal font scale for a given width."""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if new_width <= width:
            return scale/10
    return 1

def process_frame(frame):
    """Process a single frame for gender detection."""
    faces = get_faces(frame)
    male_count = 0
    female_count = 0

    for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
        face_img = frame[start_y: end_y, start_x: end_x]
        blob = cv2.dnn.blobFromImage(face_img, scalefactor=1.0, size=(227, 227), mean=MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        confidence = gender_preds[0].max()
        label = "{} - {:.2f}%".format(gender, confidence * 100)
        yPos = start_y - 15 if start_y - 15 > 15 else start_y + 15
        font_scale = get_optimal_font_scale(label, (end_x - start_x) + 25)
        box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
        cv2.putText(frame, label, (start_x, yPos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)

        # Count male and female faces
        if gender == "Male":
            male_count += 1
        else:
            female_count += 1

    # Display male and female counts on the frame
    count_text = f"Male: {male_count} | Female: {female_count}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # If female count > male count, capture image and send email
    if female_count > male_count:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = f"detected_frame_{timestamp}.jpg"
        cv2.imwrite(image_path, frame)
        send_email_with_attachment(image_path)

    return frame

def live_gender_detection():
    """Perform live gender detection using the webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Resize the frame for consistent processing
        frame = cv2.resize(frame, (640, 480))
        processed_frame = process_frame(frame)

        # Display the frame
        cv2.imshow("Live Gender Detection", processed_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    live_gender_detection()
