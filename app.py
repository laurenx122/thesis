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
    """Pre-process landmarks WITHOUT mirroring"""
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    
    if max_value == 0:
        max_value = 1
        
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

def correct_handedness(handedness_text):
    """Correct the handedness detection"""
    if handedness_text == "Right":
        return "Left"
    elif handedness_text == "Left":
        return "Right"
    return handedness_text

def logging_csv(number, landmark_list):
    """Save training data to CSV WITHOUT mirroring"""
    if 0 <= number <= 9:
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        
        # Pre-process landmarks WITHOUT mirroring
        processed_landmark_list = pre_process_landmark(landmark_list)
        
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *processed_landmark_list])
        return True
    return False

def draw_landmarks_on_image(rgb_image, detection_result, keypoint_classifier, keypoint_classifier_labels, mode, current_number):
    """Draws hand landmarks with training mode support"""
    annotated_image = np.copy(rgb_image)
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    # Create a list to store hand info for proper ordering
    hand_info = []
    
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        
        # Correct handedness
        original_handedness = handedness[0].display_name
        corrected_handedness = correct_handedness(original_handedness)
        
        # Classification (only in inference mode)
        hand_sign_label = ""
        if mode == 0:  # Inference mode
            landmark_list_for_classifier = [[lm.x, lm.y] for lm in hand_landmarks]
            pre_processed_list = pre_process_landmark(landmark_list_for_classifier)
            hand_sign_id = keypoint_classifier(pre_processed_list)
            hand_sign_label = keypoint_classifier_labels[hand_sign_id]

        hand_info.append({
            'landmarks': hand_landmarks,
            'handedness': corrected_handedness,
            'gesture': hand_sign_label,
            'original_handedness': original_handedness
        })
    
    # Sort hands: Right hand first (top), Left hand second (bottom)
    hand_info.sort(key=lambda x: x['handedness'] == "Left")
    
    # Training mode - save data from BOTH hands (NO MIRRORING)
    if mode == 1 and current_number != -1:
        for hand_data in hand_info:
            landmark_list = []
            for lm in hand_data['landmarks']:
                landmark_list.append([lm.x, lm.y])  # NO MIRRORING - save raw landmarks
            
            if logging_csv(current_number, landmark_list):
                print(f"✅ Saved {hand_data['handedness']} hand for gesture {current_number}")
    
    # Now process hands in the correct order for drawing
    for hand_idx, hand_data in enumerate(hand_info):
        hand_landmarks = hand_data['landmarks']
        handedness_text = hand_data['handedness']
        gesture_label = hand_data['gesture']

        # Drawing - Right hand at top (position 30), Left hand below (position 80)
        if handedness_text == "Right":
            text_position = (10, 30 + hand_idx * 30)
            color = (0, 255, 0)  # Green
        else:  # Left hand
            text_position = (10, 80 + hand_idx * 30)
            color = (255, 0, 0)  # Blue
        
        # Display text based on mode
        if mode == 0:  # Inference mode
            display_text = f"{handedness_text}: {gesture_label}"
        else:  # Training mode
            display_text = f"{handedness_text} - Training"
        
        cv2.putText(annotated_image, display_text, 
                    text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, color, 2, cv2.LINE_AA)
        
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
    
    # Initialize HandLandmarker
    model_path = 'hand_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Mode variables
    mode = 0  # 0: Inference, 1: Training
    current_number = -1
    frame_count = 0
    save_interval = 10  # Save every 10 frames in training mode
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎮 HAND GESTURE RECOGNITION + TRAINING")
        print("   IMPORTANT: Training WITHOUT mirroring - works for both hands!")
        print("")
        print("🎯 CONTROLS:")
        print("   K: Toggle Training/Inference mode")
        print("   0-5: Select gesture for training")
        print("   R: Reset current gesture selection") 
        print("   Q: Quit")
        print("=" * 50)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect(mp_image)
            
            # Draw results
            annotated_image = frame.copy()
            if results.hand_landmarks:
                annotated_image = draw_landmarks_on_image(
                    annotated_image, results, keypoint_classifier, keypoint_classifier_labels, 
                    mode, current_number if frame_count % save_interval == 0 else -1
                )
            
            frame_count += 1
            
            # Display mode information
            mode_text = "🎬 TRAINING MODE" if mode == 1 else "🔍 INFERENCE MODE"
            mode_color = (0, 255, 255) if mode == 1 else (255, 255, 255)
            
            cv2.putText(annotated_image, mode_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
            
            if mode == 1:
                if current_number != -1:
                    gesture_name = keypoint_classifier_labels[current_number] if current_number < len(keypoint_classifier_labels) else "Unknown"
                    cv2.putText(annotated_image, f"Training: {current_number} ({gesture_name})", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_image, "Show to BOTH hands", 
                               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(annotated_image, "Select gesture with 0-5 keys", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(annotated_image, "Press K to toggle mode, 0-5 to train, Q to quit", 
                       (10, annotated_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Hand Gesture Recognition + Training', annotated_image)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('k'):
                mode = 1 - mode
                if mode == 1:
                    print("🎬 TRAINING MODE - Press 0-5 to select gesture")
                else:
                    current_number = -1
                    print("🔍 INFERENCE MODE")
            elif key == ord('r'):
                current_number = -1
                print("🔄 Gesture selection reset")
            elif 48 <= key <= 53:
                current_number = key - 48
                gesture_name = keypoint_classifier_labels[current_number] if current_number < len(keypoint_classifier_labels) else "Unknown"
                print(f"✅ Selected gesture: {current_number} ({gesture_name})")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()