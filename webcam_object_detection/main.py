from ultralytics import YOLO
import cv2
import argparse
import supervision as sv
import numpy as np

# Define a polygon zone as a relative proportion of the frame
ZONE_POLYGON = np.array([
    [0, 0],    # Top-left corner
    [0.5, 0],  # Mid-top
    [0.5, 1],  # Mid-bottom
    [0, 1]     # Bottom-left corner
])

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for webcam resolution.
    """
    parser = argparse.ArgumentParser(description="YOLOv8 live detection with supervision")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int, help="Resolution of webcam (width height)")
    return parser.parse_args()

def initialize_camera(frame_width: int, frame_height: int):
    """
    Initialize the webcam with the specified resolution.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    return cap

def main():
    # Parse command-line arguments
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Initialize the webcam
    cap = initialize_camera(frame_width, frame_height)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load the YOLO model
    model = YOLO("yolov8l.pt")

    # Initialize annotation tools
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1)

    # Convert relative zone polygon to absolute pixel coordinates
    zone_polygon = (ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED)

    # Main loop for processing video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Perform YOLO inference
        results = model(frame, agnostic_nms=True)[0]  # Get the first result
        detections = sv.Detections.from_ultralytics(results)

        # Debugging: Print detections for inspection
        print(detections)

        # Generate labels for each detection
        labels = [
            f"{model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate the frame with bounding boxes, labels, and zone
        frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Live Detection", frame)

        # Exit the loop on pressing 'Esc'
        if cv2.waitKey(30) == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
