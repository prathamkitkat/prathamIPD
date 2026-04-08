import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ─── Initialize MediaPipe Pose ───
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def calculate_angle(a, b, c):
    """Angle at point b between vectors ba and bc. Returns degrees or None."""
    try:
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        mag_ba = np.linalg.norm(ba)
        mag_bc = np.linalg.norm(bc)
        if mag_ba == 0 or mag_bc == 0:
            return None
        cosine = np.clip(np.dot(ba, bc) / (mag_ba * mag_bc), -1.0, 1.0)
        return math.degrees(math.acos(cosine))
    except Exception:
        return None


def draw_banner(img, msg, y_pos, bg_color, h, w):
    bh = 45
    ov = img.copy()
    cv2.rectangle(ov, (10, y_pos), (w - 10, y_pos + bh), bg_color, -1)
    cv2.addWeighted(ov, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (10, y_pos), (w - 10, y_pos + bh), (255, 255, 255), 2, cv2.LINE_AA)
    ts = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
    cv2.putText(img, msg, ((w - ts[0]) // 2, y_pos + bh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def draw_form_status(img, is_good, h, w):
    label = "FORM: GOOD" if is_good else "FORM: FIX"
    bg = (0, 180, 0) if is_good else (0, 0, 200)
    ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]
    bw, bh = ts[0] + 30, ts[1] + 20
    x = (w - bw) // 2
    ov = img.copy()
    cv2.rectangle(ov, (x, 5), (x + bw, 5 + bh), bg, -1)
    cv2.addWeighted(ov, 0.8, img, 0.2, 0, img)
    cv2.rectangle(img, (x, 5), (x + bw, 5 + bh), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, label, (x + 15, 5 + bh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)


# ─── Thresholds ───
ANGLE_UP = 55         # Elbow angle = fully curled
ANGLE_DOWN = 140      # Elbow angle = fully extended
ANGLE_PARTIAL = 100   # Partial rep
UPPER_ARM_MAX_ANGLE = 35  # Max angle (degrees) upper arm can deviate from vertical


def upper_arm_angle_from_vertical(shoulder, elbow):
    """Angle between shoulder→elbow vector and straight down (vertical).
    Returns degrees. 0° = arm perfectly at side, 90° = arm horizontal."""
    # Vector from shoulder to elbow
    arm_vec = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
    # Vertical vector (straight down in image coords = positive y)
    down_vec = np.array([0, 1])
    
    mag_arm = np.linalg.norm(arm_vec)
    if mag_arm == 0:
        return 0
    
    cosine = np.clip(np.dot(arm_vec, down_vec) / mag_arm, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


class ArmTracker:
    def __init__(self, name):
        self.name = name
        self.counter = 0
        self.half_reps = 0          # Half/partial rep counter
        self.stage = 'down'
        self.locked = False
        self.is_valid_position = False
        self.max_angle_seen = 0
        self.min_angle_in_rep = 180  # Track lowest angle during current rep attempt
        self.upper_arm_deg = 0
        self.bad_position_reason = None
        self.bad_position_frames = 0  # How many consecutive frames position is bad
        self.show_bad_position = False  # Only True after 5+ bad frames
        self.started_curling = False    # True once angle starts decreasing from ANGLE_DOWN

    def update(self, elbow_angle, shoulder_xy, elbow_xy):
        if elbow_angle is None:
            self.is_valid_position = False
            self.bad_position_reason = None
            self.show_bad_position = False
            return None, False

        form_issue = None
        self.bad_position_reason = None

        # CHECK: Is the upper arm vertical? (hanging at side)
        self.upper_arm_deg = upper_arm_angle_from_vertical(shoulder_xy, elbow_xy)

        if self.upper_arm_deg > UPPER_ARM_MAX_ANGLE:
            self.bad_position_frames += 1
            self.is_valid_position = False

            # Only show feedback after 5+ consecutive bad frames (prevents flash)
            if self.bad_position_frames >= 5:
                self.show_bad_position = True
                if self.upper_arm_deg > 60:
                    self.bad_position_reason = f"{self.name}: ARM RAISED - NOT A CURL"
                else:
                    self.bad_position_reason = f"{self.name}: KEEP UPPER ARM AT YOUR SIDE"
                return self.bad_position_reason, False
            else:
                # Brief deviation — ignore, don't show feedback, don't track
                self.show_bad_position = False
                return None, True  # Still "tracking" so counter stays colored
        else:
            # Position is good — reset bad frame counter
            self.bad_position_frames = 0
            self.show_bad_position = False
            self.is_valid_position = True

        # Track angle extremes
        self.max_angle_seen = max(self.max_angle_seen, elbow_angle)

        if elbow_angle < ANGLE_UP:
            # Arm is fully curled
            self.min_angle_in_rep = min(self.min_angle_in_rep, elbow_angle)
            if self.stage == 'down' and not self.locked and self.max_angle_seen > ANGLE_DOWN:
                self.counter += 1
                self.locked = True
            self.stage = 'up'
            self.started_curling = True

        elif elbow_angle > ANGLE_DOWN:
            # Arm is fully extended — check if previous attempt was a half rep
            if self.started_curling and not self.locked and self.min_angle_in_rep > ANGLE_UP:
                # User started curling but never reached full curl
                self.half_reps += 1
                form_issue = f"{self.name}: HALF REP! CURL ALL THE WAY UP"

            self.stage = 'down'
            self.locked = False
            self.max_angle_seen = elbow_angle
            self.min_angle_in_rep = 180  # Reset for next rep
            self.started_curling = False

        else:
            # Mid-range — track minimum angle
            self.min_angle_in_rep = min(self.min_angle_in_rep, elbow_angle)

            if self.max_angle_seen > ANGLE_DOWN:
                self.started_curling = True

            # Real-time feedback: you're going back down but haven't curled enough
            if self.stage == 'up' and elbow_angle > ANGLE_PARTIAL and self.started_curling:
                form_issue = f"{self.name}: CURL HIGHER - FULL RANGE"

        return form_issue, True


if __name__ == "__main__":
    left_arm = ArmTracker("LEFT")
    right_arm = ArmTracker("RIGHT")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    prev_time = 0

    print("─────────────────────────────────────────")
    print("  Bicep Curl Tracker (Both Arms)")
    print("  Press 'q' to quit")
    print("─────────────────────────────────────────")
    print("  RULES:")
    print("  • Keep upper arm at your side (elbow below shoulder)")
    print("  • Fully extend arm down, then curl up")
    print("  • Waving/raising arms does NOT count")
    print("─────────────────────────────────────────")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_bgr.flags.writeable = True
        h, w, _ = image_bgr.shape

        form_issues = []
        left_tracking = False
        right_tracking = False

        try:
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # ─── LEFT ARM ───
                l_vis = all(lm[idx.value].visibility > 0.5 for idx in [
                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_ELBOW,
                    mp_pose.PoseLandmark.LEFT_WRIST
                ])
                if l_vis:
                    l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    l_el = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]

                    left_angle = calculate_angle(
                        [l_sh.x, l_sh.y], [l_el.x, l_el.y], [l_wr.x, l_wr.y])

                    if left_angle is not None:
                        px = (int(l_el.x * w), int(l_el.y * h))
                        cv2.putText(image_bgr, f"{int(left_angle)}",
                                    (px[0] + 10, px[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

                    issue, left_tracking = left_arm.update(
                        left_angle, [l_sh.x, l_sh.y], [l_el.x, l_el.y])
                    if issue:
                        form_issues.append((issue, (0, 0, 200) if "NOT A CURL" in issue else (0, 100, 255)))

                    # Draw indicator only if bad position persists (5+ frames)
                    if left_arm.show_bad_position and left_angle is not None:
                        el_px = (int(l_el.x * w), int(l_el.y * h))
                        sh_px = (int(l_sh.x * w), int(l_sh.y * h))
                        cv2.line(image_bgr, sh_px, el_px, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.putText(image_bgr, f"{int(left_arm.upper_arm_deg)}deg",
                                    (el_px[0] + 10, el_px[1] + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

                # ─── RIGHT ARM ───
                r_vis = all(lm[idx.value].visibility > 0.5 for idx in [
                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST
                ])
                if r_vis:
                    r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    r_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                    right_angle = calculate_angle(
                        [r_sh.x, r_sh.y], [r_el.x, r_el.y], [r_wr.x, r_wr.y])

                    if right_angle is not None:
                        px = (int(r_el.x * w), int(r_el.y * h))
                        cv2.putText(image_bgr, f"{int(right_angle)}",
                                    (px[0] - 50, px[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                    issue, right_tracking = right_arm.update(
                        right_angle, [r_sh.x, r_sh.y], [r_el.x, r_el.y])
                    if issue:
                        form_issues.append((issue, (0, 0, 200) if "NOT A CURL" in issue else (0, 100, 255)))

                    if right_arm.show_bad_position and right_angle is not None:
                        el_px = (int(r_el.x * w), int(r_el.y * h))
                        sh_px = (int(r_sh.x * w), int(r_sh.y * h))
                        cv2.line(image_bgr, sh_px, el_px, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.putText(image_bgr, f"{int(right_arm.upper_arm_deg)}deg",
                                    (el_px[0] - 60, el_px[1] + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

            else:
                cv2.putText(image_bgr, "No pose detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

        # ─── Draw Landmarks ───
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # ─── FORM STATUS ───
        draw_form_status(image_bgr, len(form_issues) == 0, h, w)

        # ─── Feedback Banners ───
        if form_issues:
            y_start = h - 55 - (len(form_issues) - 1) * 52
            for i, (msg, color) in enumerate(form_issues):
                y_pos = y_start + i * 52
                if y_pos > 0:
                    draw_banner(image_bgr, msg, y_pos, color, h, w)

        # ─── Status Bar ───
        bar_h = 90
        ov = image_bgr.copy()
        cv2.rectangle(ov, (0, 40), (w, 40 + bar_h), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.65, image_bgr, 0.35, 0, image_bgr)

        cv2.putText(image_bgr, f'FPS: {int(fps)}', (w // 2 - 40, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # LEFT arm (shown on right side because mirror)
        l_color = (50, 205, 50) if left_tracking else (100, 100, 100)
        cv2.putText(image_bgr, 'LEFT', (w - 160, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, str(left_arm.counter), (w - 160, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, l_color, 3, cv2.LINE_AA)
        stage_color = (50, 205, 50) if left_arm.stage == 'up' else (0, 165, 255)
        cv2.putText(image_bgr, left_arm.stage.upper(), (w - 80, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, stage_color, 2, cv2.LINE_AA)
        if left_arm.half_reps > 0:
            cv2.putText(image_bgr, f'HALF: {left_arm.half_reps}', (w - 160, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)

        # RIGHT arm (shown on left side because mirror)
        r_color = (0, 255, 255) if right_tracking else (100, 100, 100)
        cv2.putText(image_bgr, 'RIGHT', (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, str(right_arm.counter), (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, r_color, 3, cv2.LINE_AA)
        stage_color = (50, 205, 50) if right_arm.stage == 'up' else (0, 165, 255)
        cv2.putText(image_bgr, right_arm.stage.upper(), (100, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, stage_color, 2, cv2.LINE_AA)
        if right_arm.half_reps > 0:
            cv2.putText(image_bgr, f'HALF: {right_arm.half_reps}', (15, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)

        cv2.imshow('Bicep Curl Tracker', image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n─────────────────────────────────────────")
    print(f"  LEFT:  {left_arm.counter} reps ({left_arm.half_reps} half reps)")
    print(f"  RIGHT: {right_arm.counter} reps ({right_arm.half_reps} half reps)")
    print(f"─────────────────────────────────────────")
