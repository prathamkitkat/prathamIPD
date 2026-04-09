import cv2
from utils import get_mediapipe_pose
from process_frame_squats import ProcessFrame
from thresholds import get_thresholds_beginner, get_thresholds_pro

def main():
    # Change to get_thresholds_pro() for harder thresholds
    thresholds = get_thresholds_beginner()
    processor = ProcessFrame(thresholds=thresholds, flip_frame=True)
    pose = get_mediapipe_pose()

    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Squat Tracker running! Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, play_sound, feedback_msgs = processor.process(frame, pose)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Squat Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()
