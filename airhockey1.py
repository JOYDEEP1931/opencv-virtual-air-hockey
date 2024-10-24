import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime

# Constants
WIDTH, HEIGHT = 640, 480  #screen size
MALLET_RADIUS = 30  #radius of mallet
PUCK_RADIUS = 20  #radius of puck
SPEED_LIMIT = 15  #
GOAL_SIZE = 120   #size of the goals

# Colors
WHITE = (255, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Physics variables for puck
puck_pos = np.array([320, 240], dtype=np.float32) #position of puck
puck_vel = np.array([3, 3], dtype=np.float32)  # Starting velocity
India_score=0
England_score=0

# Function to update puck physics (movement and collision)
def update_puck():
    global puck_pos, puck_vel
    puck_pos += puck_vel

    # Collision with walls
    if puck_pos[1] - PUCK_RADIUS <= 0 or puck_pos[1] + PUCK_RADIUS >= HEIGHT:
        puck_vel[1] *= -1  # Bounce vertically
    if puck_pos[0] - PUCK_RADIUS <= 0 and (HEIGHT // 2 - GOAL_SIZE // 2) <= puck_pos[1] <= (HEIGHT // 2 + GOAL_SIZE // 2):
        India_score += 1
        reset_puck()
    elif puck_pos[0] + PUCK_RADIUS >= WIDTH and (HEIGHT // 2 - GOAL_SIZE // 2) <= puck_pos[1] <= (HEIGHT // 2 + GOAL_SIZE // 2):
        England_score += 1
        reset_puck() 
    elif puck_pos[0] - PUCK_RADIUS <= 0 or puck_pos[0] + PUCK_RADIUS >= WIDTH:
        puck_vel[0] *= -1  # Bounce horizontally    

    # Clamp speed
def reset_puck():    
    speed = np.linalg.norm(puck_vel)
    if speed > SPEED_LIMIT:
        puck_vel = (puck_vel / speed) * SPEED_LIMIT

# Function to check for collision with the mallet
def check_collision(mallet_pos):
    global puck_pos, puck_vel
    dist = np.linalg.norm(puck_pos - mallet_pos)
    if dist <= MALLET_RADIUS + PUCK_RADIUS:
        direction = (puck_pos - mallet_pos) / dist
        puck_vel = direction * SPEED_LIMIT  # Reflect puck with new velocity

# Function to control mallet with hands
def control_mallet_with_hands(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    mallet_pos = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_finger_tip = hand_landmarks.landmark[8]
            mallet_x = int(index_finger_tip.x * frame.shape[1])
            mallet_y = int(index_finger_tip.y * frame.shape[0])
            mallet_pos = np.array([mallet_x, mallet_y])
            cv2.circle(frame, (mallet_x, mallet_y), MALLET_RADIUS, RED, 60)
    return mallet_pos, frame

# Main loop for the game
duration = 120
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
start_time = datetime.now()

while True:
    ret, frame = cap.read()
    difference = (datetime.now() - start_time).seconds
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # Control mallet with hands
    mallet_pos, frame = control_mallet_with_hands(frame)
    
    # Update puck position and movement
    update_puck()
    if mallet_pos is not None:
        check_collision(mallet_pos)

    # Draw puck
    cv2.circle(frame, tuple(puck_pos.astype(int)), PUCK_RADIUS, BLUE, 30)
    
    # Draw goals
    cv2.rectangle(frame, (0, (HEIGHT // 2) - (GOAL_SIZE // 2)), (10, (HEIGHT // 2) + (GOAL_SIZE // 2)), WHITE, 20)
    cv2.rectangle(frame, (WIDTH - 10, (HEIGHT // 2) - (GOAL_SIZE // 2)), (WIDTH, (HEIGHT // 2) + (GOAL_SIZE // 2)), WHITE, 20)
    
    # Display the timer
    cv2.putText(frame, str(difference), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # Display the scores
    cv2.putText(frame, f"India : {India_score}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2,cv2.LINE_AA)
    cv2.putText(frame, f"England : {England_score}",(WIDTH-200,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2,cv2.LINE_AA)
    # Display the frame
    cv2.imshow("Air Hockey", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()