import cv2
import csv
import copy
import itertools
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

def pre_process_landmark(landmark_list):
    """Same as above"""
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

def draw_landmarks_on_image(rgb_image, detection_result, keypoint_classifier, keypoint_classifier_labels):
    """Draws hand landmarks with corrected left/right detection"""
    annotated_image = np.copy(rgb_image)
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    # Create a list to store hand info for proper ordering
    hand_info = []
    
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        
        # FIX: Swap the handedness detection
        original_handedness = handedness[0].display_name
        corrected_handedness = "Left" if original_handedness == "Right" else "Right"
        
        hand_info.append({
            'index': idx,
            'landmarks': hand_landmarks,
            'handedness': corrected_handedness,  # Use corrected handedness
            'original_handedness': original_handedness  # Keep for reference
        })
    
    # Sort hands: Right hand first (top), Left hand second (bottom)
    hand_info.sort(key=lambda x: x['handedness'] == "Left")  # Right (False) comes before Left (True)
    
    # Now process hands in the correct order
    for hand_idx, hand_data in enumerate(hand_info):
        hand_landmarks = hand_data['landmarks']
        handedness_text = hand_data['handedness']

        # Classification
        landmark_list_for_classifier = [[lm.x, lm.y] for lm in hand_landmarks]
        pre_processed_list = pre_process_landmark(landmark_list_for_classifier)
        hand_sign_id = keypoint_classifier(pre_processed_list)
        hand_sign_label = keypoint_classifier_labels[hand_sign_id]

        # Drawing - Right hand at top (position 30), Left hand below (position 80)
        if handedness_text == "Right":
            text_position = (10, 30)
        else:  # Left hand
            text_position = (10, 80)
        
        cv2.putText(annotated_image, f"{handedness_text}: {hand_sign_label}", 
                    text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
        ])
        
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
        )
            
    return annotated_image

def main():
    # Load labels and initialize classifier
    label_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    with open(label_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f) if row]
    
    keypoint_classifier = KeyPointClassifier()
    
    # Initialize HandLandmarker with IMAGE mode (synchronous)
    model_path = 'hand_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting synchronous hand gesture recognition. Press 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MP Image and process SYNCHRONOUSLY
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # This is synchronous - no callback, immediate result
            results = landmarker.detect(mp_image)
            
            # Draw results immediately
            annotated_image = frame.copy()
            if results.hand_landmarks:
                annotated_image = draw_landmarks_on_image(
                    annotated_image, results, keypoint_classifier, keypoint_classifier_labels
                )
            
            cv2.imshow('Sign Language Recognition', annotated_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()