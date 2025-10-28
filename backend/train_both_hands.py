# simple_train_both_hands.py
import cv2
import csv
import copy
import itertools
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def pre_process_landmark(landmark_list):
    """Pre-process landmarks (same as your app)"""
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

def main():
    # Load gesture labels
    label_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    with open(label_path, encoding='utf-8-sig') as f:
        gesture_labels = [row[0] for row in csv.reader(f) if row]
    
    print("🎯 Gestures:")
    for i, label in enumerate(gesture_labels):
        print(f"   {i}: {label}")

    # Initialize MediaPipe
    model_path = 'hand_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    # Training variables
    mode = 0  # 0: Normal, 1: Training
    current_number = -1
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n🎮 CONTROLS:")
        print("   K: Start/Stop training")
        print("   0-5: Select gesture")
        print("   Q: Quit")
        print("=" * 40)
        
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
            
            # Training mode - save data from BOTH hands
            if mode == 1 and current_number != -1 and results.hand_landmarks:
                for idx in range(len(results.hand_landmarks)):
                    hand_landmarks = results.hand_landmarks[idx]
                    handedness = results.handedness[idx]
                    handedness_text = handedness[0].display_name
                    
                    # Convert landmarks to list
                    landmark_list = []
                    for lm in hand_landmarks:
                        landmark_list.append([lm.x, lm.y])
                    
                    # Save to CSV
                    csv_path = 'model/keypoint_classifier/keypoint.csv'
                    processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    with open(csv_path, 'a', newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([current_number, *processed_landmark_list])
                    
                    print(f"✅ Saved {handedness_text} hand for gesture {current_number}")
            
            # Draw hand landmarks if detected
            if results.hand_landmarks:
                for idx in range(len(results.hand_landmarks)):
                    hand_landmarks = results.hand_landmarks[idx]
                    handedness = results.handedness[idx]
                    handedness_text = handedness[0].display_name
                    
                    # Draw hand info
                    color = (0, 255, 0) if handedness_text == "Right" else (255, 0, 0)
                    cv2.putText(annotated_image, f"{handedness_text}", 
                                (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, color, 2, cv2.LINE_AA)
            
            # Display UI
            mode_text = "🔴 RECORDING" if mode == 1 else "⚪ READY"
            mode_color = (0, 0, 255) if mode == 1 else (255, 255, 255)
            
            cv2.putText(annotated_image, mode_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
            
            if current_number != -1:
                gesture_name = gesture_labels[current_number] if current_number < len(gesture_labels) else "Unknown"
                cv2.putText(annotated_image, f"Gesture: {current_number} ({gesture_name})", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(annotated_image, "Press K to record, 0-5 for gesture", 
                       (10, annotated_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Simple Hand Training', annotated_image)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('k'):
                mode = 1 - mode  # Toggle mode
                if mode == 1:
                    print("🎬 STARTED recording - Show gestures to BOTH hands")
                else:
                    print("⏹️ STOPPED recording")
            elif 48 <= key <= 53:  # Numbers 0-5
                current_number = key - 48
                gesture_name = gesture_labels[current_number] if current_number < len(gesture_labels) else "Unknown"
                print(f"✅ Selected: {current_number} ({gesture_name})")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()