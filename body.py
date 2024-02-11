import cv2
import mediapipe as mp

# Initialize MediaPipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def detect_body_landmarks(image):
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Make detection
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image_rgb)
        
        # Draw landmarks on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    return image

# Read the input image file
image_file = '21.jpg'  # Change this to the path of your image file
image = cv2.imread(image_file)

# Detect body landmarks
processed_image = detect_body_landmarks(image)

# Display the output
cv2.imshow('Body Landmarks Detection', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

