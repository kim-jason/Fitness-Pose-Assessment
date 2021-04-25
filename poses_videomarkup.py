import cv2
import mediapipe as mp
import math
from threading import Thread

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# For video input:
input_video_name = '/Users/jasonkim/Downloads/Virtual Fitness Test.mov' #Update file path to appropriate name
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(input_video_name)

# Check that video file is correctly opened
if cap.isOpened() == False:
    print("Error reading video file")
    quit() 

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

output_video_name = '/Users/jasonkim/Downloads/Virtual Fitness Test_MARKED.mov' #Update file path the appropriate name

fps = cap.get(cv2.CAP_PROP_FPS)

result = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'jpeg'), fps, size)
result.set(cv2.CAP_PROP_BUFFERSIZE, 2)


while True:

    success, image = cap.read()
        
    if not success:
        print("Either an error or got to the end of video")
        quit()

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    image_height, image_width, _ = image.shape

    left_hip = -9999
    left_knee = -9999
    left_ankle = -9999

    right_hip = -9999
    right_knee = -9999
    right_ankle = -9999

    left_hip_to_knee = -9999
    left_knee_to_ankle = -9999
    left_hip_to_ankle = -9999

    right_hip_to_knee = -9999
    right_knee_to_ankle = -9999
    right_hip_to_ankle = -9999

    left_knee_angle = 9999
    right_knee_angle = 9999

    landmarks = results.pose_landmarks

    if landmarks:
        left_hip = landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
        left_knee = landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE]
        right_hip = landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]
        right_knee = landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE]

        left_hip_to_knee = math.sqrt((left_knee.x*image_width - left_hip.x*image_width)**2 + (left_knee.y*image_height - left_hip.y*image_height)**2)
        left_knee_to_ankle = math.sqrt((left_ankle.x*image_width - left_knee.x*image_width)**2 + (left_ankle.y*image_height - left_knee.y*image_height)**2)
        left_hip_to_ankle = math.sqrt((left_ankle.x*image_width - left_hip.x*image_width)**2 + (left_ankle.y*image_height - left_hip.y*image_height)**2)

        right_hip_to_knee = math.sqrt((right_knee.x*image_width - right_hip.x*image_width)**2 + (right_knee.y*image_height - right_hip.y*image_height)**2)
        right_knee_to_ankle = math.sqrt((right_ankle.x*image_width - right_knee.x*image_width)**2 + (right_ankle.y*image_height - right_knee.y*image_height)**2)
        right_hip_to_ankle = math.sqrt((right_ankle.x*image_width - right_hip.x*image_width)**2 + (right_ankle.y*image_height - right_hip.y*image_height)**2)

        left_knee_angle = math.degrees(math.acos((left_hip_to_knee**2 + left_knee_to_ankle**2 - left_hip_to_ankle**2)/(2*left_hip_to_knee*left_knee_to_ankle)))
        right_knee_angle = math.degrees(math.acos((right_hip_to_knee**2 + right_knee_to_ankle**2 - right_hip_to_ankle**2)/(2*right_hip_to_knee*right_knee_to_ankle)))

    angles_text = "Left and Right Knee Angles: " + str(int(left_knee_angle)) + " | " + str(int(right_knee_angle))
    reps_text = "Number Squat Reps Done: " + str(int(num_reps))

    color_text=(0,0,255)
    if left_knee_angle < 80 and right_knee_angle < 80:
        color_text = (0,255,0)
        if went_back_up == True:
            num_reps += 1
            went_back_up = False
    else:
        went_back_up = True
    cv2.putText(img=image, text=angles_text, org=(10,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.75, color=color_text, thickness=3)
    cv2.putText(img=image, text=reps_text, org=(10,120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.50, color=(255,0,0), thickness=2)

    result.write(image)
    #cv2.imshow('MediaPipe Pose', image)
    key = cv2.waitKey(int(1/fps * 1000))

    if key == ord('q'):
        cv2.destroyAllWindows()
        print("User stopped the program in middle of processing")
        break

pose.close()
cap.release()
result.release()

print("Video successfully saved")