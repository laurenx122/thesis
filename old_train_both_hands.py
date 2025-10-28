import cv2
import csv
import copy
import itertools
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

def pre_process_landmark(landmark_list, is_right_hand=True):
    """Pre-process landmarks with hand-specific normalization"""
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # MIRROR landmarks for LEFT hand to make them consistent with RIGHT hand training
    if not is_right_hand:
        for i in range(len(temp_landmark_list)):
            temp_landmark_list[i][0] = -temp_landmark_list[i][0]  # Mirror x-coordinate

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    
    if max_value == 0:
        max_value = 1
        
    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

def correct_handedness(handedness_text):
    """CORRECT the handedness detection - swap Left/Right"""
    if handedness_text == "Right":
        return "Left"
    elif handedness_text == "Left":
        return "Right"
    return handedness_text

def draw_landmarks_on_image(rgb_image, detection_result, mode, current_number):
    """Draw landmarks with CORRECTED handedness"""
    annotated_image = np.copy(rgb_image)
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        original_handedness = handedness[0].display_name
        corrected_handedness = correct_handedness(original_handedness)

        # Draw hand information with CORRECTED labels
        color = (0, 255, 0) if corrected_handedness == "Right" else (255, 0, 0)
        cv2.putText(annotated_image, f"{corrected_handedness} (was {original_handedness})", 
                    (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2, cv2.LINE_AA)
        
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

def logging_csv(number, landmark_list, handedness):
    """Save training data to CSV with CORRECTED handedness"""
    if 0 <= number <= 9:
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        
        # CORRECT the handedness
        corrected_handedness = correct_handedness(handedness)
        is_right_hand = (corrected_handedness == "Right")
        
        # Pre-process landmarks with hand-specific processing
        processed_landmark_list = pre_process_landmark(landmark_list, is_right_hand)
        
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            # Save: number, *landmark_data (NO hand flag to maintain compatibility)
            writer.writerow([number, *processed_landmark_list])
        print(f"✅ Saved: Gesture {number} for {corrected_handedness} hand (was {handedness})")

def main():
    # Load gesture labels
    label_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    try:
        with open(label_path, encoding='utf-8-sig') as f:
            gesture_labels = [row[0] for row in csv.reader(f) if row]
        print("✅ Loaded gesture labels:")
        for i, label in enumerate(gesture_labels):
            print(f"   {i}: {label}")
    except FileNotFoundError:
        print("❌ Error: Could not find keypoint_classifier_label.csv")
        return

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
    frame_count = 0
    save_interval = 8  # Save every 8 frames to avoid duplicates
    
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            print("\n🎯 TRAINING CONTROLS:")
            print("   K: Enter training mode")
            print("   0-9: Select gesture number")
            print("   N: Exit training mode") 
            print("   Q: Quit")
            print("=" * 50)
            print("📝 IMPORTANT: Handedness is now CORRECTED")
            print("   - Your RIGHT hand will be labeled 'Right'")
            print("   - Your LEFT hand will be labeled 'Left'")
            print("=" * 50)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)  # Mirror for selfie view
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = landmarker.detect(mp_image)
                
                # Draw results
                annotated_image = frame.copy()
                if results.hand_landmarks:
                    annotated_image = draw_landmarks_on_image(annotated_image, results, mode, current_number)
                    
                    # Training mode - save data
                    if mode == 1 and current_number != -1 and frame_count % save_interval == 0:
                        for idx in range(len(results.hand_landmarks)):
                            hand_landmarks = results.hand_landmarks[idx]
                            handedness = results.handedness[idx]
                            handedness_text = handedness[0].display_name
                            
                            # Convert landmarks to list
                            landmark_list = []
                            for lm in hand_landmarks:
                                landmark_list.append([lm.x, lm.y])
                            
                            logging_csv(current_number, landmark_list, handedness_text)
                
                frame_count += 1
                
                # Display UI information
                mode_text = "🎬 TRAINING MODE" if mode == 1 else "📹 NORMAL MODE"
                mode_color = (0, 255, 255) if mode == 1 else (255, 255, 255)
                
                cv2.putText(annotated_image, mode_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
                
                if current_number != -1:
                    gesture_name = gesture_labels[current_number] if current_number < len(gesture_labels) else "Unknown"
                    cv2.putText(annotated_image, f"Gesture: {current_number} ({gesture_name})", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_image, "Show to BOTH hands", 
                               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show correction info
                cv2.putText(annotated_image, "Handedness: AUTO-CORRECTED", 
                           (10, annotated_image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(annotated_image, "Press 'K' to train, '0-9' for gesture, 'N' to stop", 
                           (10, annotated_image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Hand Gesture Training - CORRECTED HANDEDNESS', annotated_image)
                
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('k'):  # Training mode
                    mode = 1
                    print("🎬 TRAINING MODE ACTIVATED")
                    print("   Show gestures to BOTH hands - Handedness is auto-corrected")
                elif key == ord('n'):  # Normal mode
                    mode = 0
                    current_number = -1
                    print("⏹️  Training stopped")
                elif 48 <= key <= 57:  # Numbers 0-9
                    current_number = key - 48
                    gesture_name = gesture_labels[current_number] if current_number < len(gesture_labels) else "Unknown"
                    print(f"✅ Selected gesture: {current_number} ({gesture_name})")
                    if mode == 1:
                        print("   → Show this gesture to BOTH hands")

            cap.release()
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure hand_landmarker.task is in the root directory")

if __name__ == "__main__":
    main()