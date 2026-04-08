import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line


class ProcessFrame:
    def __init__(self, thresholds, flip_frame = False):
        
        # Set if frame should be flipped or not.
        self.flip_frame = flip_frame

        # self.thresholds
        self.thresholds = thresholds

        # Font type.
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # line type
        self.linetype = cv2.LINE_AA

        # set radius to draw arc
        self.radius = 20

        # Colors in BGR format.
        self.COLORS = {
                        'blue'       : (0, 127, 255),
                        'red'        : (255, 50, 50),
                        'green'      : (0, 255, 127),
                        'light_green': (100, 233, 127),
                        'yellow'     : (255, 255, 0),
                        'magenta'    : (255, 0, 255),
                        'white'      : (255,255,255),
                        'cyan'       : (0, 255, 255),
                        'light_blue' : (102, 204, 255)
                      }



        # Dictionary to maintain the various landmark features.
        self.dict_features = {}
        self.left_features = {
                                'shoulder': 11,
                                'elbow'   : 13,
                                'wrist'   : 15,                    
                                'hip'     : 23,
                                'knee'    : 25,
                                'ankle'   : 27,
                                'foot'    : 31
                             }

        self.right_features = {
                                'shoulder': 12,
                                'elbow'   : 14,
                                'wrist'   : 16,
                                'hip'     : 24,
                                'knee'    : 26,
                                'ankle'   : 28,
                                'foot'    : 32
                              }

        self.dict_features['left'] = self.left_features
        self.dict_features['right'] = self.right_features
        self.dict_features['nose'] = 0

        
        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            'state_seq': [],

            'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,

            # 0 --> Bend Backwards, 1 --> Bend Forward, 2 --> Keep shin straight, 3 --> Deep squat
            'DISPLAY_TEXT' : np.full((4,), False),
            'COUNT_FRAMES' : np.zeros((4,), dtype=np.int64),

            'LOWER_HIPS': False,

            'INCORRECT_POSTURE': False,

            'prev_state': None,
            'curr_state':None,

            'SQUAT_COUNT': 0,
            'IMPROPER_SQUAT':0
            
        }
        
        self.FEEDBACK_ID_MAP = {
                                0: ('BACK NOT UPRIGHT - STRAIGHTEN UP', 215, (0, 100, 255)),
                                1: ('LEAN TORSO FORWARD SLIGHTLY', 215, (0, 100, 255)),
                                2: ('KNEES PAST TOES - SIT BACK', 170, (0, 0, 220)),
                                3: ('TOO DEEP - COME UP A BIT', 125, (0, 0, 220))
                               }

        


    def _get_state(self, knee_angle):
        
        knee = None        

        if self.thresholds['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['NORMAL'][1]:
            knee = 1
        elif self.thresholds['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['TRANS'][1]:
            knee = 2
        elif self.thresholds['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= self.thresholds['HIP_KNEE_VERT']['PASS'][1]:
            knee = 3

        return f's{knee}' if knee else None



    
    def _update_state_sequence(self, state):

        if state == 's2':
            if (('s3' not in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2'))==0) or \
                    (('s3' in self.state_tracker['state_seq']) and (self.state_tracker['state_seq'].count('s2')==1)):
                        self.state_tracker['state_seq'].append(state)
            

        elif state == 's3':
            if (state not in self.state_tracker['state_seq']) and 's2' in self.state_tracker['state_seq']: 
                self.state_tracker['state_seq'].append(state)

            


    def _draw_feedback_banner(self, frame, msg, y_pos, bg_color, text_color=(255, 255, 255)):
        """Draw a large, prominent feedback banner across the frame."""
        frame_h, frame_w = frame.shape[:2]
        banner_h = 50
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, y_pos), (frame_w - 10, y_pos + banner_h), bg_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border
        cv2.rectangle(frame, (10, y_pos), (frame_w - 10, y_pos + banner_h), (255, 255, 255), 2, lineType=self.linetype)
        
        # Text centered
        text_size = cv2.getTextSize(msg, self.font, 0.9, 2)[0]
        text_x = (frame_w - text_size[0]) // 2
        text_y = y_pos + banner_h - 15
        cv2.putText(frame, msg, (text_x, text_y), self.font, 0.9, text_color, 2, lineType=self.linetype)
        
        return frame


    def _draw_form_status(self, frame, is_good_form):
        """Draw a live FORM STATUS indicator at top-center."""
        frame_h, frame_w = frame.shape[:2]
        
        if is_good_form:
            label = "FORM: GOOD"
            bg = (0, 200, 0)
        else:
            label = "FORM: FIX"
            bg = (0, 0, 220)
        
        text_size = cv2.getTextSize(label, self.font, 1.0, 3)[0]
        box_w = text_size[0] + 30
        box_h = text_size[1] + 24
        x_start = (frame_w - box_w) // 2
        y_start = 5
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + box_w, y_start + box_h), bg, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x_start, y_start), (x_start + box_w, y_start + box_h), (255, 255, 255), 2, lineType=self.linetype)
        
        cv2.putText(frame, label, (x_start + 15, y_start + box_h - 10), self.font, 1.0, (255, 255, 255), 3, lineType=self.linetype)
        
        return frame


    def _show_feedback(self, frame, c_frame, dict_maps, lower_hips_disp):

        active_feedback = []

        if lower_hips_disp:
            active_feedback.append(('>> GO DEEPER - LOWER YOUR HIPS <<', (0, 180, 255)))

        for idx in np.where(c_frame)[0]:
            active_feedback.append((dict_maps[idx][0], dict_maps[idx][2]))

        # Print feedback to terminal too
        if active_feedback:
            msgs = [f[0] for f in active_feedback]
            print(f"\r[FORM] {' | '.join(msgs)}        ", end='', flush=True)

        # Draw each feedback as a large banner at the bottom of the frame
        frame_h = frame.shape[0]
        y_start = frame_h - 60 - (len(active_feedback) - 1) * 58 if active_feedback else frame_h

        for i, (msg, color) in enumerate(active_feedback):
            y_pos = y_start + i * 58
            if y_pos > 0:
                frame = self._draw_feedback_banner(frame, msg, y_pos, color)

        return frame



    def process(self, frame: np.array, pose):
        play_sound = None
       

        frame_height, frame_width, _ = frame.shape

        # Process the image.
        keypoints = pose.process(frame)

        if keypoints.pose_landmarks:
            ps_lm = keypoints.pose_landmarks

            nose_coord = get_landmark_features(ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
            left_shldr_coord, left_elbow_coord, left_wrist_coord, left_hip_coord, left_knee_coord, left_ankle_coord, left_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
            right_shldr_coord, right_elbow_coord, right_wrist_coord, right_hip_coord, right_knee_coord, right_ankle_coord, right_foot_coord = \
                                get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

            offset_angle = find_angle(left_shldr_coord, right_shldr_coord, nose_coord)

            if offset_angle > self.thresholds['OFFSET_THRESH']: 
                
                display_inactivity = False

                end_time = time.perf_counter()
                self.state_tracker['INACTIVE_TIME_FRONT'] += end_time - self.state_tracker['start_inactive_time_front']
                self.state_tracker['start_inactive_time_front'] = end_time

                if self.state_tracker['INACTIVE_TIME_FRONT'] >= self.thresholds['INACTIVE_THRESH']:
                    self.state_tracker['SQUAT_COUNT'] = 0
                    self.state_tracker['IMPROPER_SQUAT'] = 0
                    display_inactivity = True

                cv2.circle(frame, nose_coord, 7, self.COLORS['white'], -1)
                cv2.circle(frame, left_shldr_coord, 7, self.COLORS['yellow'], -1)
                cv2.circle(frame, right_shldr_coord, 7, self.COLORS['magenta'], -1)

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)

                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                    self.state_tracker['start_inactive_time_front'] = time.perf_counter()

                draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  
                

                draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                    
                )  
                
                # Big warning banner for camera alignment
                frame = self._draw_feedback_banner(frame, 'CAMERA NOT ALIGNED! STAND SIDEWAYS', 
                                                   frame_height - 120, (0, 100, 255))
                
                draw_text(
                    frame, 
                    'OFFSET ANGLE: '+str(offset_angle), 
                    pos=(30, frame_height-30),
                    text_color=(255, 255, 230),
                    font_scale=0.65,
                    text_color_bg=(255, 153, 0),
                ) 

                # Reset inactive times for side view.
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
                self.state_tracker['prev_state'] =  None
                self.state_tracker['curr_state'] = None
            
            # Camera is aligned properly.
            else:

                self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
                self.state_tracker['start_inactive_time_front'] = time.perf_counter()


                dist_l_sh_hip = abs(left_foot_coord[1]- left_shldr_coord[1])
                dist_r_sh_hip = abs(right_foot_coord[1] - right_shldr_coord[1])

                shldr_coord = None
                elbow_coord = None
                wrist_coord = None
                hip_coord = None
                knee_coord = None
                ankle_coord = None
                foot_coord = None

                if dist_l_sh_hip > dist_r_sh_hip:
                    shldr_coord = left_shldr_coord
                    elbow_coord = left_elbow_coord
                    wrist_coord = left_wrist_coord
                    hip_coord = left_hip_coord
                    knee_coord = left_knee_coord
                    ankle_coord = left_ankle_coord
                    foot_coord = left_foot_coord

                    multiplier = -1
                                     
                
                else:
                    shldr_coord = right_shldr_coord
                    elbow_coord = right_elbow_coord
                    wrist_coord = right_wrist_coord
                    hip_coord = right_hip_coord
                    knee_coord = right_knee_coord
                    ankle_coord = right_ankle_coord
                    foot_coord = right_foot_coord

                    multiplier = 1
                    

                # ------------------- Vertical Angle calculation --------------
                
                hip_vertical_angle = find_angle(shldr_coord, np.array([hip_coord[0], 0]), hip_coord)
                cv2.ellipse(frame, hip_coord, (30, 30), 
                            angle = 0, startAngle = -90, endAngle = -90+multiplier*hip_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3, lineType = self.linetype)

                draw_dotted_line(frame, hip_coord, start=hip_coord[1]-80, end=hip_coord[1]+20, line_color=self.COLORS['blue'])




                knee_vertical_angle = find_angle(hip_coord, np.array([knee_coord[0], 0]), knee_coord)
                cv2.ellipse(frame, knee_coord, (20, 20), 
                            angle = 0, startAngle = -90, endAngle = -90-multiplier*knee_vertical_angle, 
                            color = self.COLORS['white'], thickness = 3,  lineType = self.linetype)

                draw_dotted_line(frame, knee_coord, start=knee_coord[1]-50, end=knee_coord[1]+20, line_color=self.COLORS['blue'])



                ankle_vertical_angle = find_angle(knee_coord, np.array([ankle_coord[0], 0]), ankle_coord)
                cv2.ellipse(frame, ankle_coord, (30, 30),
                            angle = 0, startAngle = -90, endAngle = -90 + multiplier*ankle_vertical_angle,
                            color = self.COLORS['white'], thickness = 3,  lineType=self.linetype)

                draw_dotted_line(frame, ankle_coord, start=ankle_coord[1]-50, end=ankle_coord[1]+20, line_color=self.COLORS['blue'])

                # ------------------------------------------------------------
        
                
                # Join landmarks — color them based on form quality  
                # Default: light blue = good, red = bad joint
                knee_color = self.COLORS['light_blue']
                ankle_color = self.COLORS['light_blue']
                hip_color = self.COLORS['light_blue']

                cv2.line(frame, shldr_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, wrist_coord, elbow_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, shldr_coord, hip_coord, self.COLORS['light_blue'], 4, lineType=self.linetype)
                cv2.line(frame, knee_coord, hip_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, knee_coord,self.COLORS['light_blue'], 4,  lineType=self.linetype)
                cv2.line(frame, ankle_coord, foot_coord, self.COLORS['light_blue'], 4,  lineType=self.linetype)
                
                # Plot landmark points
                cv2.circle(frame, shldr_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, elbow_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, wrist_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, hip_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, knee_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, ankle_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)
                cv2.circle(frame, foot_coord, 7, self.COLORS['yellow'], -1,  lineType=self.linetype)

                

                current_state = self._get_state(int(knee_vertical_angle))
                self.state_tracker['curr_state'] = current_state
                self._update_state_sequence(current_state)



                # -------------------------------------- COMPUTE COUNTERS --------------------------------------

                if current_state == 's1':

                    if len(self.state_tracker['state_seq']) == 3 and not self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['SQUAT_COUNT']+=1
                        play_sound = str(self.state_tracker['SQUAT_COUNT'])
                        
                    elif 's2' in self.state_tracker['state_seq'] and len(self.state_tracker['state_seq'])==1:
                        self.state_tracker['IMPROPER_SQUAT']+=1
                        play_sound = 'incorrect'

                    elif self.state_tracker['INCORRECT_POSTURE']:
                        self.state_tracker['IMPROPER_SQUAT']+=1
                        play_sound = 'incorrect'
                        
                    
                    self.state_tracker['state_seq'] = []
                    self.state_tracker['INCORRECT_POSTURE'] = False

                    # CLEAR all feedback immediately when standing up
                    self.state_tracker['DISPLAY_TEXT'] = np.full((4,), False)
                    self.state_tracker['COUNT_FRAMES'] = np.zeros((4,), dtype=np.int64)
                    self.state_tracker['LOWER_HIPS'] = False


                # ----------------------------------------------------------------------------------------------------




                # -------------------------------------- PERFORM FEEDBACK ACTIONS --------------------------------------

                else:
                    if hip_vertical_angle > self.thresholds['HIP_THRESH'][1]:
                        self.state_tracker['DISPLAY_TEXT'][0] = True
                        # Clear contradictory: can't be "too upright" AND "too forward"
                        self.state_tracker['DISPLAY_TEXT'][1] = False
                        self.state_tracker['COUNT_FRAMES'][1] = 0
                        

                    elif hip_vertical_angle < self.thresholds['HIP_THRESH'][0] and \
                         self.state_tracker['state_seq'].count('s2')==1:
                            self.state_tracker['DISPLAY_TEXT'][1] = True
                            # Clear contradictory
                            self.state_tracker['DISPLAY_TEXT'][0] = False
                            self.state_tracker['COUNT_FRAMES'][0] = 0
                        
                                        
                    
                    if self.thresholds['KNEE_THRESH'][0] < knee_vertical_angle < self.thresholds['KNEE_THRESH'][1] and \
                       self.state_tracker['state_seq'].count('s2')==1:
                        self.state_tracker['LOWER_HIPS'] = True


                    elif knee_vertical_angle > self.thresholds['KNEE_THRESH'][2]:
                        self.state_tracker['DISPLAY_TEXT'][3] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True

                    
                    if (ankle_vertical_angle > self.thresholds['ANKLE_THRESH']):
                        self.state_tracker['DISPLAY_TEXT'][2] = True
                        self.state_tracker['INCORRECT_POSTURE'] = True


                # ----------------------------------------------------------------------------------------------------


                
                
                # ----------------------------------- COMPUTE INACTIVITY ---------------------------------------------

                display_inactivity = False
                
                if self.state_tracker['curr_state'] == self.state_tracker['prev_state']:

                    end_time = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']
                    self.state_tracker['start_inactive_time'] = end_time

                    if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                        self.state_tracker['SQUAT_COUNT'] = 0
                        self.state_tracker['IMPROPER_SQUAT'] = 0
                        display_inactivity = True

                
                else:
                    
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                # -------------------------------------------------------------------------------------------------------
              


                hip_text_coord_x = hip_coord[0] + 10
                knee_text_coord_x = knee_coord[0] + 15
                ankle_text_coord_x = ankle_coord[0] + 10

                if self.flip_frame:
                    frame = cv2.flip(frame, 1)
                    hip_text_coord_x = frame_width - hip_coord[0] + 10
                    knee_text_coord_x = frame_width - knee_coord[0] + 15
                    ankle_text_coord_x = frame_width - ankle_coord[0] + 10

                
                
                if 's3' in self.state_tracker['state_seq'] or current_state == 's1':
                    self.state_tracker['LOWER_HIPS'] = False

                # Suppress contradictory feedback:
                # Don't say "GO DEEPER" if knees are past toes (problem is forward travel, not depth)
                # Don't say "GO DEEPER" if already too deep
                if self.state_tracker['DISPLAY_TEXT'][2] or self.state_tracker['DISPLAY_TEXT'][3]:
                    self.state_tracker['LOWER_HIPS'] = False

                self.state_tracker['COUNT_FRAMES'][self.state_tracker['DISPLAY_TEXT']]+=1

                # ---------- ENHANCED FEEDBACK RENDERING ----------

                # Determine if form is currently good
                has_errors = np.any(self.state_tracker['DISPLAY_TEXT']) or self.state_tracker['LOWER_HIPS']
                frame = self._draw_form_status(frame, not has_errors)

                # Draw large feedback banners at the bottom
                frame = self._show_feedback(frame, self.state_tracker['COUNT_FRAMES'], self.FEEDBACK_ID_MAP, self.state_tracker['LOWER_HIPS'])

                # Color-code the problematic joints — highlight in RED
                if self.state_tracker['DISPLAY_TEXT'][2]:  # Knee over toe
                    # Draw thick red circle around ankle+knee to highlight the issue
                    if self.flip_frame:
                        ankle_draw = (frame_width - ankle_coord[0], ankle_coord[1])
                        knee_draw = (frame_width - knee_coord[0], knee_coord[1])
                    else:
                        ankle_draw = tuple(ankle_coord)
                        knee_draw = tuple(knee_coord)
                    cv2.circle(frame, ankle_draw, 18, (0, 0, 255), 3, lineType=self.linetype)
                    cv2.circle(frame, knee_draw, 18, (0, 0, 255), 3, lineType=self.linetype)

                if self.state_tracker['DISPLAY_TEXT'][0] or self.state_tracker['DISPLAY_TEXT'][1]:  # Hip angle issues
                    if self.flip_frame:
                        hip_draw = (frame_width - hip_coord[0], hip_coord[1])
                    else:
                        hip_draw = tuple(hip_coord)
                    cv2.circle(frame, hip_draw, 18, (0, 0, 255), 3, lineType=self.linetype)

                if self.state_tracker['DISPLAY_TEXT'][3]:  # Squat too deep
                    if self.flip_frame:
                        knee_draw = (frame_width - knee_coord[0], knee_coord[1])
                    else:
                        knee_draw = tuple(knee_coord)
                    cv2.circle(frame, knee_draw, 22, (0, 0, 255), 4, lineType=self.linetype)

                # -------------------------------------------------


                if display_inactivity:
                    play_sound = 'reset_counters'
                    self.state_tracker['start_inactive_time'] = time.perf_counter()
                    self.state_tracker['INACTIVE_TIME'] = 0.0

                
                cv2.putText(frame, str(int(hip_vertical_angle)), (hip_text_coord_x, hip_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(knee_vertical_angle)), (knee_text_coord_x, knee_coord[1]+10), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)
                cv2.putText(frame, str(int(ankle_vertical_angle)), (ankle_text_coord_x, ankle_coord[1]), self.font, 0.6, self.COLORS['light_green'], 2, lineType=self.linetype)

                 
                draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  
                

                draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                    
                )  
                
                
                self.state_tracker['DISPLAY_TEXT'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = False
                self.state_tracker['COUNT_FRAMES'][self.state_tracker['COUNT_FRAMES'] > self.thresholds['CNT_FRAME_THRESH']] = 0    
                self.state_tracker['prev_state'] = current_state
                                  

       
        
        else:

            if self.flip_frame:
                frame = cv2.flip(frame, 1)

            end_time = time.perf_counter()
            self.state_tracker['INACTIVE_TIME'] += end_time - self.state_tracker['start_inactive_time']

            display_inactivity = False

            if self.state_tracker['INACTIVE_TIME'] >= self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0
                display_inactivity = True

            self.state_tracker['start_inactive_time'] = end_time

            draw_text(
                    frame, 
                    "CORRECT: " + str(self.state_tracker['SQUAT_COUNT']), 
                    pos=(int(frame_width*0.68), 30),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(18, 185, 0)
                )  
                

            draw_text(
                    frame, 
                    "INCORRECT: " + str(self.state_tracker['IMPROPER_SQUAT']), 
                    pos=(int(frame_width*0.68), 80),
                    text_color=(255, 255, 230),
                    font_scale=0.7,
                    text_color_bg=(221, 0, 0),
                    
                )  

            if display_inactivity:
                play_sound = 'reset_counters'
                self.state_tracker['start_inactive_time'] = time.perf_counter()
                self.state_tracker['INACTIVE_TIME'] = 0.0
            
            
            # Reset all other state variables
            
            self.state_tracker['prev_state'] =  None
            self.state_tracker['curr_state'] = None
            self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0
            self.state_tracker['INCORRECT_POSTURE'] = False
            self.state_tracker['DISPLAY_TEXT'] = np.full((4,), False)
            self.state_tracker['COUNT_FRAMES'] = np.zeros((4,), dtype=np.int64)
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()
            
            
            
        return frame, play_sound