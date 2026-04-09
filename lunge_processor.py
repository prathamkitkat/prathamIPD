import cv2
import time
import numpy as np
import math

def calculate_angle_3pt(a, b, c):
    """Calculate angle between points A, B, C with B as vertex"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

class ProcessFrameLunge:
    def __init__(self, flip_frame=True):
        self.flip_frame = flip_frame
        self.lunge_count = 0
        self.improper_lunge = 0
        self.stage = 'up'
        self.last_active_time = time.time()
        self.inactive_thresh = 10.0
        
        self.form_issues = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def process(self, frame, pose):
        if self.flip_frame:
            frame = cv2.flip(frame, 1)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_bgr.flags.writeable = True
        h, w, _ = image_bgr.shape

        self.form_issues = []

        try:
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                
                # Check visibility
                joints = [11, 12, 23, 24, 25, 26, 27, 28, 0] # shoulders, hips, knees, ankles, nose
                visible = all(lm[i].visibility > 0.5 for i in joints)
                
                if visible:
                    self.last_active_time = time.time()
                    
                    # Compute Z to see which side is closer
                    l_sh = lm[11]
                    r_sh = lm[12]
                    
                    is_left_visible = l_sh.z < r_sh.z
                    
                    # Coordinates
                    nose = (int(lm[0].x * w), int(lm[0].y * h))
                    
                    l_hip = (int(lm[23].x * w), int(lm[23].y * h))
                    r_hip = (int(lm[24].x * w), int(lm[24].y * h))
                    l_knee = (int(lm[25].x * w), int(lm[25].y * h))
                    r_knee = (int(lm[26].x * w), int(lm[26].y * h))
                    l_ankle = (int(lm[27].x * w), int(lm[27].y * h))
                    r_ankle = (int(lm[28].x * w), int(lm[28].y * h))
                    l_foot_index = (int(lm[31].x * w), int(lm[31].y * h))
                    r_foot_index = (int(lm[32].x * w), int(lm[32].y * h))
                    
                    # Torso coordinates
                    l_sh_pt = (int(l_sh.x * w), int(l_sh.y * h))
                    r_sh_pt = (int(r_sh.x * w), int(r_sh.y * h))

                    # Use the side that is closer to camera for torso
                    hip_pt = l_hip if is_left_visible else r_hip
                    sh_pt = l_sh_pt if is_left_visible else r_sh_pt
                    
                    # Determine facing direction: > 0 means facing right, < 0 means facing left
                    dir_val = nose[0] - sh_pt[0]
                    dir_sign = 1 if dir_val > 0 else -1
                    
                    # Determine front leg
                    l_ankle_rel = dir_sign * l_ankle[0]
                    r_ankle_rel = dir_sign * r_ankle[0]
                    
                    if l_ankle_rel > r_ankle_rel:
                        front_hip, front_knee, front_ankle, front_foot = l_hip, l_knee, l_ankle, l_foot_index
                        back_hip, back_knee, back_ankle = r_hip, r_knee, r_ankle
                    else:
                        front_hip, front_knee, front_ankle, front_foot = r_hip, r_knee, r_ankle, r_foot_index
                        back_hip, back_knee, back_ankle = l_hip, l_knee, l_ankle
                        
                    # Calculate angles
                    front_knee_angle = calculate_angle_3pt(front_hip, front_knee, front_ankle)
                    back_knee_angle = calculate_angle_3pt(back_hip, back_knee, back_ankle)
                    torso_angle = calculate_angle_3pt(sh_pt, hip_pt, (hip_pt[0], hip_pt[1] - 100)) # vertical line
                    
                    if not hasattr(self, 'has_error_this_rep'):
                        self.has_error_this_rep = False
                    if not hasattr(self, 'reached_depth'):
                        self.reached_depth = False

                    # Identify if camera is sideways enough
                    offset = min(abs(l_sh_pt[0] - r_sh_pt[0]), abs(l_hip[0] - r_hip[0]))
                    if offset > w * 0.15:  # threshold for side profile
                         self.form_issues.append("CAMERA NOT ALIGNED! STAND SIDEWAYS")
                         self.has_error_this_rep = True

                    # Draw skeleton
                    cv2.line(image_bgr, sh_pt, hip_pt, (255, 255, 255), 4)
                    
                    cv2.line(image_bgr, front_hip, front_knee, (0, 255, 255), 4)
                    cv2.line(image_bgr, front_knee, front_ankle, (0, 255, 255), 4)
                    cv2.line(image_bgr, front_ankle, front_foot, (0, 255, 255), 4)
                    
                    cv2.line(image_bgr, back_hip, back_knee, (255, 0, 255), 4)
                    cv2.line(image_bgr, back_knee, back_ankle, (255, 0, 255), 4)
                    
                    for pt in [sh_pt, hip_pt, front_knee, front_ankle, back_knee, back_ankle]:
                        cv2.circle(image_bgr, pt, 6, (0, 0, 255), -1)
                        
                    cv2.putText(image_bgr, f"{int(front_knee_angle)}", (front_knee[0] + 10, front_knee[1] - 10), self.font, 0.6, (0, 255, 255), 2)
                    cv2.putText(image_bgr, f"{int(back_knee_angle)}", (back_knee[0] + 10, back_knee[1] - 10), self.font, 0.6, (255, 0, 255), 2)

                    # --- Form Processing ---
                    # Torso Lean check
                    if torso_angle > 25:
                        self.form_issues.append("KEEP TORSO UPRIGHT")
                        self.has_error_this_rep = True
                        
                    # Knee over toe check (front knee)
                    knee_rel_foot = dir_sign * (front_knee[0] - front_foot[0])
                    if knee_rel_foot > 10 and front_knee_angle < 130:
                        self.form_issues.append("FRONT KNEE PAST TOES - WIDEN STANCE")
                        self.has_error_this_rep = True

                    # Back knee depth check
                    if self.stage == 'down':
                        if back_knee_angle <= 110:
                            self.reached_depth = True
                        elif not getattr(self, 'reached_depth', False):
                            self.form_issues.append("LOWER YOUR BACK KNEE MORE")

                    # --- State Machine ---
                    if front_knee_angle > 150 and back_knee_angle > 150:
                        if self.stage == 'down':
                            if not getattr(self, 'reached_depth', False) or getattr(self, 'has_error_this_rep', False):
                                self.improper_lunge += 1
                            else:
                                self.lunge_count += 1
                        self.stage = 'up'
                        self.has_error_this_rep = False
                        self.reached_depth = False
                    elif front_knee_angle < 110:
                        if self.stage == 'up':
                            self.has_error_this_rep = False
                            self.reached_depth = False
                        self.stage = 'down'
                    
                else:
                    cv2.putText(image_bgr, "Ensure full body is visible", (50, 50), self.font, 0.7, (0, 165, 255), 2)
                    
                # Reset counters if inactive
                if (time.time() - self.last_active_time) > self.inactive_thresh:
                    self.stage = 'up'
                    self.lunge_count = 0
                    self.improper_lunge = 0
                    self.last_active_time = time.time()
                    
        except Exception as e:
            print(f"Error in lunge processing: {e}")

        # Render overlays
        bar_h = 70
        ov = image_bgr.copy()
        cv2.rectangle(ov, (0, 40), (w, 40 + bar_h), (20, 20, 20), -1)
        cv2.addWeighted(ov, 0.65, image_bgr, 0.35, 0, image_bgr)
        
        form_color = (0, 200, 0) if len(self.form_issues) == 0 else (0, 0, 255)
        
        cv2.putText(image_bgr, 'LUNGES', (15, 65), self.font, 0.7, (200, 200, 200), 1)
        cv2.putText(image_bgr, str(self.lunge_count), (15, 100), self.font, 1.5, form_color, 3)
        cv2.putText(image_bgr, self.stage.upper(), (100, 100), self.font, 0.8, (255, 255, 255), 2)
        
        y_start = h - 55 - (len(self.form_issues) - 1) * 52
        for i, msg in enumerate(self.form_issues):
            y_pos = y_start + i * 52
            if y_pos > 0:
                bh = 45
                ov = image_bgr.copy()
                cv2.rectangle(ov, (10, y_pos), (w - 10, y_pos + bh), (0, 100, 255), -1)
                cv2.addWeighted(ov, 0.85, image_bgr, 0.15, 0, image_bgr)
                cv2.putText(image_bgr, msg, (20, y_pos + bh - 12), self.font, 0.75, (255, 255, 255), 2)
                
        return image_bgr, self.form_issues
