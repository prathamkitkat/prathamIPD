import cv2
from run_curl import ArmTracker, calculate_angle, draw_form_status, draw_banner, mp_pose, mp_drawing, mp_drawing_styles

class ProcessFrameBicep:
    def __init__(self, flip_frame=True):
        self.flip_frame = flip_frame
        self.left_arm = ArmTracker("LEFT")
        self.right_arm = ArmTracker("RIGHT")

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

                    issue, left_tracking = self.left_arm.update(
                        left_angle, [l_sh.x, l_sh.y], [l_el.x, l_el.y])
                    if issue:
                        form_issues.append((issue, (0, 0, 200) if "NOT A CURL" in issue else (0, 100, 255)))

                    if self.left_arm.show_bad_position and left_angle is not None:
                        el_px = (int(l_el.x * w), int(l_el.y * h))
                        sh_px = (int(l_sh.x * w), int(l_sh.y * h))
                        cv2.line(image_bgr, sh_px, el_px, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.putText(image_bgr, f"{int(self.left_arm.upper_arm_deg)}deg",
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

                    issue, right_tracking = self.right_arm.update(
                        right_angle, [r_sh.x, r_sh.y], [r_el.x, r_el.y])
                    if issue:
                        form_issues.append((issue, (0, 0, 200) if "NOT A CURL" in issue else (0, 100, 255)))

                    if self.right_arm.show_bad_position and right_angle is not None:
                        el_px = (int(r_el.x * w), int(r_el.y * h))
                        sh_px = (int(r_sh.x * w), int(r_sh.y * h))
                        cv2.line(image_bgr, sh_px, el_px, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.putText(image_bgr, f"{int(self.right_arm.upper_arm_deg)}deg",
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

        # LEFT arm
        l_color = (50, 205, 50) if left_tracking else (100, 100, 100)
        cv2.putText(image_bgr, 'LEFT', (w - 160, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, str(self.left_arm.counter), (w - 160, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, l_color, 3, cv2.LINE_AA)
        stage_color = (50, 205, 50) if self.left_arm.stage == 'up' else (0, 165, 255)
        cv2.putText(image_bgr, self.left_arm.stage.upper(), (w - 80, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, stage_color, 2, cv2.LINE_AA)
        if self.left_arm.half_reps > 0:
            cv2.putText(image_bgr, f'HALF: {self.left_arm.half_reps}', (w - 160, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)

        # RIGHT arm
        r_color = (0, 255, 255) if right_tracking else (100, 100, 100)
        cv2.putText(image_bgr, 'RIGHT', (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image_bgr, str(self.right_arm.counter), (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, r_color, 3, cv2.LINE_AA)
        stage_color = (50, 205, 50) if self.right_arm.stage == 'up' else (0, 165, 255)
        cv2.putText(image_bgr, self.right_arm.stage.upper(), (100, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, stage_color, 2, cv2.LINE_AA)
        if self.right_arm.half_reps > 0:
            cv2.putText(image_bgr, f'HALF: {self.right_arm.half_reps}', (15, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1, cv2.LINE_AA)

        return image_bgr, None
