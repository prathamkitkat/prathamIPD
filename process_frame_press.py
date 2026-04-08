import cv2
import time
from run_press import (
    PressArmTracker, calculate_angle, draw_form_status, draw_banner,
    mp_pose, mp_drawing, mp_drawing_styles,
    SYMMETRY_THRESH
)
from thresholds import get_thresholds_beginner


class ProcessFramePress:
    """
    Processes individual video frames for shoulder-press rep counting and
    form correction.  Unified rep counter — both arms must go up and come
    down together for one rep.

    Interface:
        process(frame, pose) -> (processed_frame, feedback_list)
    """

    def __init__(self, flip_frame=True):
        self.flip_frame = flip_frame
        self.left_arm = PressArmTracker("LEFT")
        self.right_arm = PressArmTracker("RIGHT")
        self.last_active_time = time.time()
        self.last_left_angle = None
        self.last_right_angle = None
        self.thresholds = get_thresholds_beginner()

        # Unified rep counting
        self.press_counter = 0
        self.been_at_top = False

    def process(self, frame, pose):
        if self.flip_frame:
            frame = cv2.flip(frame, 1)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_bgr.flags.writeable = True
        h, w, _ = image_bgr.shape

        form_issues = []
        left_tracking = False
        right_tracking = False
        left_angle = None
        right_angle = None

        try:
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # ─── Left Arm (MediaPipe RIGHT because of mirror flip) ───
                l_vis = all(lm[idx.value].visibility > 0.5 for idx in [
                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.RIGHT_ELBOW,
                    mp_pose.PoseLandmark.RIGHT_WRIST
                ])
                if l_vis:
                    l_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    l_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    l_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                    left_angle = calculate_angle(
                        [l_sh.x, l_sh.y], [l_el.x, l_el.y], [l_wr.x, l_wr.y])

                    if left_angle is not None:
                        px = (int(l_el.x * w), int(l_el.y * h))
                        cv2.putText(image_bgr, f"{int(left_angle)}",
                                    (px[0] + 10, px[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

                    issue, left_tracking = self.left_arm.update(
                        left_angle, [l_sh.x, l_sh.y], [l_el.x, l_el.y], [l_wr.x, l_wr.y])
                    if issue:
                        form_issues.append((issue, (0, 0, 200) if "RAISE ARMS" in issue else (0, 100, 255)))

                    if self.left_arm.show_bad_position and left_angle is not None:
                        el_px = (int(l_el.x * w), int(l_el.y * h))
                        sh_px = (int(l_sh.x * w), int(l_sh.y * h))
                        cv2.line(image_bgr, sh_px, el_px, (0, 0, 255), 3, cv2.LINE_AA)

                # ─── Right Arm (MediaPipe LEFT because of mirror flip) ───
                r_vis = all(lm[idx.value].visibility > 0.5 for idx in [
                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_ELBOW,
                    mp_pose.PoseLandmark.LEFT_WRIST
                ])
                if r_vis:
                    r_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    r_el = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    r_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]

                    right_angle = calculate_angle(
                        [r_sh.x, r_sh.y], [r_el.x, r_el.y], [r_wr.x, r_wr.y])

                    if right_angle is not None:
                        px = (int(r_el.x * w), int(r_el.y * h))
                        cv2.putText(image_bgr, f"{int(right_angle)}",
                                    (px[0] - 50, px[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                    issue, right_tracking = self.right_arm.update(
                        right_angle, [r_sh.x, r_sh.y], [r_el.x, r_el.y], [r_wr.x, r_wr.y])
                    if issue:
                        form_issues.append((issue, (0, 0, 200) if "RAISE ARMS" in issue else (0, 100, 255)))

                    if self.right_arm.show_bad_position and right_angle is not None:
                        el_px = (int(r_el.x * w), int(r_el.y * h))
                        sh_px = (int(r_sh.x * w), int(r_sh.y * h))
                        cv2.line(image_bgr, sh_px, el_px, (0, 0, 255), 3, cv2.LINE_AA)

                # ─── UNIFIED REP COUNTING ───
                both_up = self.left_arm.stage == 'up' and self.right_arm.stage == 'up'
                both_down = self.left_arm.stage == 'down' and self.right_arm.stage == 'down'

                if both_up:
                    self.been_at_top = True

                if both_down and self.been_at_top:
                    self.press_counter += 1
                    self.been_at_top = False

                # ─── SYMMETRY + FLARE CHECKS ───
                if l_vis and r_vis:
                    l_wr_lm = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    r_wr_lm = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    wrist_y_diff = abs(l_wr_lm.y - r_wr_lm.y)
                    if wrist_y_diff > SYMMETRY_THRESH and \
                       (self.left_arm.started_pressing or self.right_arm.started_pressing):
                        form_issues.append(("UNEVEN - PRESS BOTH ARMS TOGETHER", (0, 140, 255)))



            else:
                cv2.putText(image_bgr, "No pose detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {e}")

        # ─── INACTIVITY CHECK ───
        is_active = False
        if results.pose_landmarks:
            if left_angle is not None:
                if self.last_left_angle is None or abs(left_angle - self.last_left_angle) > 5:
                    is_active = True
                self.last_left_angle = left_angle

            if right_angle is not None:
                if self.last_right_angle is None or abs(right_angle - self.last_right_angle) > 5:
                    is_active = True
                self.last_right_angle = right_angle

        if is_active:
            self.last_active_time = time.time()
        else:
            if (time.time() - self.last_active_time) >= self.thresholds['INACTIVE_THRESH']:
                self.press_counter = 0
                self.been_at_top = False
                self.left_arm.stage = 'down'
                self.left_arm.started_pressing = False
                self.right_arm.stage = 'down'
                self.right_arm.started_pressing = False
                self.last_active_time = time.time()

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

        stage_label = 'UP' if (self.left_arm.stage == 'up' and self.right_arm.stage == 'up') else 'DOWN'
        counter_color = (50, 205, 50) if (left_tracking and right_tracking) else (100, 100, 100)
        stage_color = (50, 205, 50) if stage_label == 'UP' else (0, 165, 255)

        cv2.putText(image_bgr, 'PRESS', (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, str(self.press_counter), (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, counter_color, 3, cv2.LINE_AA)
        cv2.putText(image_bgr, stage_label, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, stage_color, 2, cv2.LINE_AA)

        # Return feedback messages (strip colour tuples)
        return image_bgr, [msg for msg, _ in form_issues]
