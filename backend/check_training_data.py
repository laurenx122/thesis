# start_fresh.py
import csv
import os

def start_fresh():
    """Create a minimal valid training dataset"""
    output_path = 'model/keypoint_classifier/keypoint.csv'
    
    # Create minimal valid data for testing
    # Format: [gesture_number, landmark1_x, landmark1_y, landmark2_x, landmark2_y, ...]
    minimal_data = []
    
    # Add a few samples for each gesture (0-5)
    for gesture in range(6):  # 0=Open, 1=Close, 2=Pointer, 3=OK, 4=Hello, 5=Peace
        for _ in range(10):  # 10 samples per gesture
            # Create dummy landmark data (42 zeros)
            landmarks = [0.0] * 42
            row = [gesture] + landmarks
            minimal_data.append(row)
    
    # Save fresh data
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(minimal_data)
    
    print(f"✅ Created fresh training data:")
    print(f"   Total samples: {len(minimal_data)}")
    print(f"   Elements per row: {len(minimal_data[0])}")
    print(f"   Samples per gesture: 10")
    print(f"   You can now retrain and then collect proper data")

if __name__ == "__main__":
    start_fresh()