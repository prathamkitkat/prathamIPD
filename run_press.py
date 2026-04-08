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


# ─── Shoulder Press Thresholds ───
ANGLE_TOP = 140        # Elbow angle = arms extended overhead (lowered — camera can't see full 160)
ANGLE_BOTTOM = 90      # Elbow angle = elbows bent at bottom
ANGLE_PARTIAL = 120    # Partial lockout threshold

# Wrist alignment: wrist should stay roughly above elbow
WRIST_DRIFT_THRESH = 0.06

# Symmetry: both wrists should be at roughly the same height
SYMMETRY_THRESH = 0.10



# Upper arm must be raised at least this many degrees from vertical to count
MIN_UPPER_ARM_ELEVATION = 55


def upper_arm_angle_from_vertical(shoulder, elbow):
    """Angle between shoulder→elbow vector and straight down.
    0° = arm hanging, 90° = horizontal, 180° = overhead."""
    arm_vec = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
    down_vec = np.array([0, 1])
    mag_arm = np.linalg.norm(arm_vec)
    if mag_arm == 0:
        return 0
    cosine = np.clip(np.dot(arm_vec, down_vec) / mag_arm, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


class PressArmTracker:
    """Tracks a single arm's state and form issues for shoulder press.
    Does NOT count reps — that is done at a unified level."""

    def __init__(self, name):
        self.name = name
        self.stage = 'down'        # 'down' = elbows bent, 'up' = arms extended
        self.is_valid_position = False
        self.upper_arm_deg = 0
        self.elbow_angle = None
        self.bad_position_reason = None
        self.bad_position_frames = 0
        self.show_bad_position = False
        self.started_pressing = False

    def update(self, elbow_angle, shoulder_xy, elbow_xy, wrist_xy):
        """
        Update arm state. Returns (form_issue_or_None, is_tracking).
        Stage transitions (up/down) are tracked here; rep counting is external.
        """
        if elbow_angle is None:
            self.is_valid_position = False
            self.bad_position_reason = None
            self.show_bad_position = False
            return None, False

        self.elbow_angle = elbow_angle
        form_issue = None
        self.bad_position_reason = None

        # CHECK: Upper arm elevated enough?
        self.upper_arm_deg = upper_arm_angle_from_vertical(shoulder_xy, elbow_xy)

        if self.upper_arm_deg < MIN_UPPER_ARM_ELEVATION:
            self.bad_position_frames += 1
            self.is_valid_position = False
            if self.bad_position_frames >= 5:
                self.show_bad_position = True
                self.bad_position_reason = f"{self.name}: RAISE ARMS HIGHER"
                return self.bad_position_reason, False
            else:
                self.show_bad_position = False
                return None, True
        else:
            self.bad_position_frames = 0
            self.show_bad_position = False
            self.is_valid_position = True

        # CHECK: Wrist drift
        wrist_drift = abs(wrist_xy[0] - elbow_xy[0])
        if wrist_drift > WRIST_DRIFT_THRESH and self.started_pressing:
            form_issue = f"{self.name}: KEEP FOREARM VERTICAL"

        # Stage transitions
        if elbow_angle >= ANGLE_TOP:
            self.stage = 'up'
            self.started_pressing = True
        elif elbow_angle <= ANGLE_BOTTOM:
            self.stage = 'down'
            self.started_pressing = True
        # else: mid-range, stage stays as-is

        return form_issue, True


if __name__ == "__main__":
    left_arm = PressArmTracker("LEFT")
    right_arm = PressArmTracker("RIGHT")

    # ─── Unified rep counter ───
    press_counter = 0
    half_reps = 0
    unified_stage = 'down'  # tracks the combined state
    been_at_top = False      # True once both arms reach 'up' in this rep cycle

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    prev_time = 0

    print("─────────────────────────────────────────")
    print("  Shoulder Press Tracker (Unified)")
    print("  Press 'q' to quit")
    print("─────────────────────────────────────────")
    print("  RULES:")
    print("  • Raise both arms to shoulder height")
    print("  • Press both arms fully overhead together")
    print("  • Lower back to ~90° elbows together")
    print("  • Keep wrists above elbows")
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
                left_angle = None
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
                        left_angle, [l_sh.x, l_sh.y], [l_el.x, l_el.y], [l_wr.x, l_wr.y])
                    if issue:
                        form_issues.append((issue, (0, 0, 200) if "RAISE ARMS" in issue else (0, 100, 255)))

                    if left_arm.show_bad_position and left_angle is not None:
                        el_px = (int(l_el.x * w), int(l_el.y * h))
                        sh_px = (int(l_sh.x * w), int(l_sh.y * h))
                        cv2.line(image_bgr, sh_px, el_px, (0, 0, 255), 3, cv2.LINE_AA)

                # ─── RIGHT ARM ───
                r_vis = all(lm[idx.value].visibility > 0.5 for idx in [
                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST
                ])
                right_angle = None
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
                        right_angle, [r_sh.x, r_sh.y], [r_el.x, r_el.y], [r_wr.x, r_wr.y])
                    if issue:
                        form_issues.append((issue, (0, 0, 200) if "RAISE ARMS" in issue else (0, 100, 255)))

                    if right_arm.show_bad_position and right_angle is not None:
                        el_px = (int(r_el.x * w), int(r_el.y * h))
                        sh_px = (int(r_sh.x * w), int(r_sh.y * h))
                        cv2.line(image_bgr, sh_px, el_px, (0, 0, 255), 3, cv2.LINE_AA)

                # ─── UNIFIED REP COUNTING ───
                both_up = left_arm.stage == 'up' and right_arm.stage == 'up'
                both_down = left_arm.stage == 'down' and right_arm.stage == 'down'

                if both_up:
                    been_at_top = True

                if both_down and been_at_top:
                    press_counter += 1
                    been_at_top = False

                # ─── SYMMETRY CHECK (both arms visible) ───
                if l_vis and r_vis:
                    wrist_y_diff = abs(l_wr.y - r_wr.y)
                    if wrist_y_diff > SYMMETRY_THRESH and \
                       (left_arm.started_pressing or right_arm.started_pressing):
                        form_issues.append(("UNEVEN - PRESS BOTH ARMS TOGETHER", (0, 140, 255)))



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

        # ─── Status Bar (unified) ───
        bar_h = 70
        ov = image_bgr.copy()
        cv2.rectangle(ov, (0, 40), (w, 40 + bar_h), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.65, image_bgr, 0.35, 0, image_bgr)

        cv2.putText(image_bgr, f'FPS: {int(fps)}', (w - 120, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        # Unified counter — centered
        stage_label = 'UP' if (left_arm.stage == 'up' and right_arm.stage == 'up') else 'DOWN'
        counter_color = (50, 205, 50) if (left_tracking and right_tracking) else (100, 100, 100)
        stage_color = (50, 205, 50) if stage_label == 'UP' else (0, 165, 255)

        cv2.putText(image_bgr, 'PRESS', (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, str(press_counter), (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, counter_color, 3, cv2.LINE_AA)
        cv2.putText(image_bgr, stage_label, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, stage_color, 2, cv2.LINE_AA)

        cv2.imshow('Shoulder Press Tracker', image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n─────────────────────────────────────────")
    print(f"  SHOULDER PRESS: {press_counter} reps")
    print(f"─────────────────────────────────────────")
