import cv2
import json
from datetime import datetime
from ultralytics import YOLO

# Load a lightweight, pre-trained model (simulating the Edge NPU capabilities)
# YOLOv8 Nano is small enough to run on a standard laptop CPU in real-time.
model = YOLO('yolov8n.pt') 

# Initialize laptop webcam (simulating the factory industrial camera)
cap = cv2.VideoCapture(0)

# This list will act as our "Semantic Compression" payload
shift_log_payload = []

print("Starting Edge Node Surveillance... Press 'q' to stop and generate JSON.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run lightweight object detection on the current frame
    results = model(frame, verbose=False)
    
    current_time = datetime.now().isoformat()
    frame_events = []

    # Extract spatial data and discard the raw image data
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract bounding box coordinates, confidence, and class name
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # Only log confident detections to save bandwidth
            if conf > 0.50:
                event = {
                    "timestamp": current_time,
                    "object_class": class_name,
                    "confidence": round(conf, 2),
                    "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                }
                frame_events.append(event)

    # If anything was detected in this frame, add it to our transmission payload
    if frame_events:
        shift_log_payload.extend(frame_events)

    # Display the feed strictly for debugging the prototype 
    # (In production, the edge device runs headless with no display)
    annotated_frame = results[0].plot()
    cv2.imshow("Edge Node - NPU Simulation", annotated_frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up hardware resources (Destroying video data)
cap.release()
cv2.destroyAllWindows()

# Output the Semantic Compression Payload
payload_filename = "semantic_payload.json"
with open(payload_filename, "w") as json_file:
    json.dump(shift_log_payload, json_file, indent=4)

print(f"\nSurveillance ended. Raw video destroyed.")
print(f"Privacy-preserving JSON payload saved to {payload_filename}. Ready for Cloud transmission.")