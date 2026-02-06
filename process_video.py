import cv2
from ultralytics import YOLO
import os
from collections import defaultdict  # Tool for counting

# --- CONFIGURATION ---
model_path = 'models/YOLOv8_Small_RDD.pt'
output_folder = 'output_results'

def process_video_with_report(video_path):
    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    print(f"🚀 Loading model from {model_path}...")
    model = YOLO(model_path)

    # 2. Setup Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    # Video Properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. Setup Writer (Forces MP4)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    base_name = os.path.basename(video_path)
    save_path = os.path.join(output_folder, f"{os.path.splitext(base_name)[0]}_detected.mp4")
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # --- 4. INITIALIZE COUNTER ---
    # This dictionary will auto-create keys like {'Pothole': 0}
    detection_stats = defaultdict(int)

    print(f"🎥 Processing: {base_name} ({total_frames} frames)")
    print("Press 'q' in the preview window to stop early.\n")

    frame_count = 0
    
    # 5. The Loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Run YOLO
        # stream=True makes it faster/smoother for videos
        results = model(frame, conf=0.25, verbose=False)
        
        # --- COUNTING LOGIC ---
        # Look at every box found in this single frame
        for box in results[0].boxes:
            # Get the ID (e.g., 0)
            cls_id = int(box.cls[0])
            # Get the Name (e.g., 'D40' or 'Pothole') using the model's internal list
            cls_name = model.names[cls_id]
            
            # Add to total
            detection_stats[cls_name] += 1
        # ----------------------

        # Draw boxes and Write Frame
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # Show Preview
        cv2.imshow("Processing...", annotated_frame)
        
        # Progress indicator (every 50 frames)
        if frame_count % 50 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...", end='\r')

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n🛑 Stopped early by user.")
            break

    # 6. Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # --- 7. PRINT THE REPORT ---
    print("\n" + "="*40)
    print("📊 FINAL DETECTION REPORT")
    print("="*40)
    
    if not detection_stats:
        print("✅ No road damage detected.")
    else:
        # Sort them so the most frequent damage appears first
        sorted_stats = sorted(detection_stats.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'TYPE':<20} | {'COUNT':<10}")
        print("-" * 33)
        for label, count in sorted_stats:
            print(f"{label:<20} | {count:<10}")
            
    print("="*40)
    print(f"💾 Video saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    target = input("Drag and drop video here: ").strip().replace('"', '')
    if os.path.exists(target):
        process_video_with_report(target)
    else:
        print("❌ File not found.")