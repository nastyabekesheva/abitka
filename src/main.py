import cv2
import mediapipe as mp
from clothing_overlay import overlay_png, overlay_logo


# Initialize MediaPipe Face Mesh and Pose
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_pose = mp.solutions.pose.Pose()

cap = cv2.VideoCapture(0)

def to_pixel(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_res = mp_face.process(rgb)
    pose_res = mp_pose.process(rgb)

    head_top = None
    if face_res.multi_face_landmarks:

        # Get all needed points
        face = face_res.multi_face_landmarks[0]
        get = lambda i: to_pixel(face.landmark[i], w, h)

        chin = get(152)
        forehead = get(10)
        left_ear = get(234)
        right_ear = get(454)
        left_eye = get(33)
        right_eye = get(263)

        # Head width and height
        head_width = ((right_ear[0] - left_ear[0])**2 + (right_ear[1] - left_ear[1])**2)**0.5
        head_height = chin[1] - forehead[1]

        # Eye center for horizontal alignment
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        # Improved top estimate: move UP from forehead by head_height * factor
        HEAD_TOP_FACTOR = 0.35
        top_y = forehead[1] - int(head_height * HEAD_TOP_FACTOR)
        top_x = eye_center[0]
        head_top = (top_x, top_y)


    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark

        l_hip = to_pixel(lm[mp.solutions.pose.PoseLandmark.LEFT_HIP], w, h)
        r_hip = to_pixel(lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP], w, h)
        skirt_pos = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2 + 20)

        # Load skirt image and resize
        skirt_img = cv2.imread("/Users/nastyabekesheva/Projects/abitka/abitka/data/skirt_flipped.png", cv2.IMREAD_UNCHANGED)
        skirt_scale = 0.7
        skirt_resized = cv2.resize(skirt_img, (0, 0), fx=skirt_scale, fy=skirt_scale)
        skirt_h, skirt_w = skirt_resized.shape[:2]

        # Load tail image and resize
        tail_img = cv2.imread("/Users/nastyabekesheva/Projects/abitka/abitka/data/хвіст.png", cv2.IMREAD_UNCHANGED)
        tail_scale = 0.65
        tail_resized = cv2.resize(tail_img, (0, 0), fx=tail_scale, fy=tail_scale)
        tail_h, tail_w = tail_resized.shape[:2]

        # Calculate skirt top-left corner
        skirt_top_left = (skirt_pos[0] - skirt_w // 2, skirt_pos[1] - skirt_h // 2)

        # Tail position: align tail top-left to skirt top-left + small right offset (e.g., +15 px)
        offset_x = 15
        tail_pos = (skirt_top_left[0] + 20, skirt_top_left[1] + 210)

        # Overlay skirt and tail
        frame = overlay_png(frame, "/Users/nastyabekesheva/Projects/abitka/abitka/data/хвіст.png", tail_pos, scale=tail_scale)
        frame = overlay_png(frame, "/Users/nastyabekesheva/Projects/abitka/abitka/data/skirt_flipped.png", skirt_pos, scale=skirt_scale)
        

    if head_top:
        frame = overlay_png(frame, "/Users/nastyabekesheva/Projects/abitka/abitka/data/ears_wide.png", head_top, scale=0.5)

    frame = overlay_logo(frame, "/Users/nastyabekesheva/Projects/abitka/abitka/data/nastya_skin_photo_s_ushkami.png", opacity=0.7)

    cv2.imshow("Funny AR Try-On", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
