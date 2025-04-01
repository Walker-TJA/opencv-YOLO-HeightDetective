import pyttsx3 # Text-to-speech library
import cv2 as cv # OpenCV for camera access
from ultralytics import YOLO #ai architecture
import time
import threading # For threading to run TTS in the background

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the ai
model = YOLO('yolov8n-pose.pt')  # Replace with the correct model if needed

# Open the camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Define the reference values from my height at 10 feet away
reference_pixels = 320  # Person height in pixels at 5'7"-8"
reference_height = 5.75  # 5'7"-8" in feet (averaged)

# Variable to track the last time the height was spoken (not really sure how it works haha)
last_speak_time = time.time()

# To store the bounding box and track the person
track_person = None

# Define a function for text-to-speech for seperate thread
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Function to convert feet to feet and inches (sepearates the values)
def feet_to_inches(feet):
    inches = feet * 12
    feet_int = int(inches // 12)  # Extract the feet
    inches_rem = round(inches % 12)  # Get the remaining inches
    return feet_int, inches_rem
speak_text("Please stand 10 feet from camera, place the camera at 3 feet above ground, and make sure camera is facing straight on at a perfect 90 degrees.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Run the model on the frame (including keypoints for body tracking)
    results = model(frame)

    # Filter out only people (pretrained yolo can detect many things not jsut people)
    person_detections = []
    for result in results:
        # Extract the class IDs and check if they correspond to 'person' (class ID 0)
        cls = result.boxes.cls  # Tensor containing class IDs
        #no idea what the above 2 lines are. i had to ask ai
        for i, c in enumerate(cls):
            if c.item() == 0:  # Check if it's a person (class 0)
                person_detections.append(result[i])  # Add the detection to the list

    if person_detections: #NOTE: I had an ideo of how the math works but i had to ask ai for help. CALCULUS SUCKS!!!
        # Track the first person detected (focus on one person)
        track_person = person_detections[0]
        bbox = track_person.boxes.xyxy[0]  # Get the bounding box coordinates (x1, y1, x2, y2)
        
        # Get the pixel height (difference between y-coordinates)
        pixel_height = int(bbox[3] - bbox[1])  # Height in pixels (bbox equals the corners of the bounding box around the person)
        
        # Calculate the real-world height using the reference values
        real_height = (pixel_height / reference_pixels) * reference_height
        
        # Convert the height from decimal feet to feet and inches
        feet, inches = feet_to_inches(real_height)
        
        # Draw the bounding box on the frame
        x1, y1, x2, y2 = map(int, bbox)  # Convert to integers for drawing
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw keypoints (if available) for body pose tracking
        if track_person.keypoints is not None:
            for keypoint in track_person.keypoints[0]:  # Loop through each keypoint
                # Ensure keypoint has the proper number of values
                if len(keypoint) == 3:
                    x, y, confidence = keypoint
                    if confidence > 0.3:  # Only draw keypoints with high confidence
                        cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                else:
                    print(f"Unexpected keypoint format: {keypoint}")
        
        # Display the real-world height on the frame
        cv.putText(frame, f"Height: {feet} feet {inches} inches", 
                   (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Speak height every 7 seconds
        if time.time() - last_speak_time >= 7:
            speak_thread = threading.Thread(target=speak_text, args=(f"Person is about {feet} feet {inches} inches tall.",))
            speak_thread.start()  # Start the thread to speak the height
            last_speak_time = time.time()

    # Display the resulting frame
    cv.imshow('Detected Frame', frame)

    # Exit the loop when the 'q' key is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv.destroyAllWindows()
