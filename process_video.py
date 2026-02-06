import cv2
from ultralytics import YOLO
import os

# --- CONFIGURATION ---
model_path = 'models/YOLOv8_Small_RDD.pt'   # Double check this filename!
output_folder = 'output_results'

def process_video_to_mp4(video_path):
    # 1. Load Model
    if not os.path.exists(model_path):
        print(f" Error: Model not found at {model_path}")
        return
    model = YOLO(model_path)

    # 2. Setup Video Capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Error: Could not open video.")
        return

    # Get video properties (width, height, fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 3. Setup Video Writer (The Part that forces MP4)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create the output filename
    base_name = os.path.basename(video_path)
    name_no_ext = os.path.splitext(base_name)[0]
    save_path = os.path.join(output_folder, f"{name_no_ext}_detected.mp4")

    # 'mp4v' is the standard MP4 codec for Windows/OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    print(f"Processing: {base_name}")
    print(f" Saving to: {save_path}")
    print("Press 'q' in the preview window to stop early.")

    # 4. The Loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO on the frame
        results = model(frame, conf=0.25, verbose=False)
        
        # Draw the boxes on the frame
        annotated_frame = results[0].plot()

        # Write the frame to the new MP4 file
        out.write(annotated_frame)

        # Optional: Show a preview window while processing
        cv2.imshow("Processing... (Press Q to stop)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 5. Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\nDone! Your MP4 file is ready.")
    print(f" Location: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    target = input("Drag and drop video here: ").strip().replace('"', '')
    if os.path.exists(target):
        process_video_to_mp4(target)
    else:
        print(" File not found.")