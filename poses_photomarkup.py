

import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


# For static images:
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

file_name = '/Users/jasonkim/Downloads/raw.jpg' #Update file path to appropriate name
image = cv2.imread(file_name)
image_height, image_width, _ = image.shape

# Convert the BGR image to RGB before processing.
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

print(
    f'Nose coordinates: ('
    f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
    f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
)
# Draw pose landmarks on the image.
annotated_image = image.copy()
mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
cv2.imwrite('/Users/jasonkim/Downloads/' + 'raw_MARKUP' + '.jpg', annotated_image) #Update file path to appropriate name
pose.close()