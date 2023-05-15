import cv2
import mediapipe as mp
from imutils import face_utils
import dlib

CIRCLE_RADIUS = 50
CIRCLE_DISTANCE = 140
CIRCLE_TOUCH_COLOR = (0, 0, 255)
FINGERTIP_RADIUS = 0
ANIMATION_LENGTH = 120
IMG_SQUARE_SIDE = 1.4*CIRCLE_RADIUS
PROJECT_FOLDER = '/Users/EvanChen/project/facial/'

current_menu = 'NONE'
menus = {'EYES': {'menu_count': 3}}
button_num = 0

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(255, 128, 0), thickness=6)
handConStyle = mpDraw.DrawingSpec(color=(176, 224, 230), thickness=6)
finger_pos = [0, 0]

LANDMARKS = {}


def addfilter(image, x_offset, y_offset, imgurl, resize_x, resize_y):

    filter_image = cv2.imread(
        imgurl, cv2.IMREAD_COLOR)
    if imgurl == "/Users/EvanChen/project/facial/facial/nose/nose1.jpg":
        resize_y = 300
    filter_image_resized = cv2.resize(filter_image, (resize_x, resize_y))
    x_end = x_offset + filter_image_resized.shape[1]
    y_end = y_offset + filter_image_resized.shape[0]
    roi = image[y_offset:y_end, x_offset:x_end]

    filter_image_gray = cv2.cvtColor(
        filter_image_resized, cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(
        filter_image_gray, 120, 255, cv2.THRESH_BINARY)
    bg = cv2.bitwise_or(roi, roi, mask=mask)
    mask_inv = cv2.bitwise_not(filter_image_gray)
    fg = cv2.bitwise_and(
        filter_image_resized, filter_image_resized, mask=mask_inv)
    final_roi = cv2.add(bg, fg)
    image[y_offset:y_end, x_offset:x_end] = final_roi
    return image


def drawIcon(icon, center, image):
    new_img = image.copy()
    icon = cv2.imread(icon)

    new_img[center[1]-35:center[1]-35+70,
            center[0]-35:center[0]+35] = icon
    return new_img


def DrawFace(img):
    img = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for n, (x, y) in enumerate(shape):
            LANDMARKS[n+1] = [x, y]
    return img


def DrawMenu(img, frame):
    init_pos_y = 75
    image = img.copy()
    global current_menu

    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand_need_result = hands.process(imgRGB)
    if hand_need_result.multi_hand_landmarks:
        for handLms in hand_need_result.multi_hand_landmarks:

            for i, lm in enumerate(handLms.landmark):
                xPos = int(lm.x * imgWidth)
                yPos = int(lm.y * imgHeight)

                if i == 8 and xPos > 0 and yPos > 0:
                    finger_pos = [xPos, yPos]
                    current_menu = detectnowmenu(finger_pos)

    else:
        current_menu = 'NONE'
    if current_menu != 'NONE':
        if(frame < 30):
            for i in range(4):
                imagelink = PROJECT_FOLDER + 'facial/icons/icon_' + \
                    'EYES'+'_'+str(i)+'_'+'off.png'
                image = cv2.circle(
                    image, [1150, init_pos_y+i*CIRCLE_DISTANCE-90*i+frame*3*i], CIRCLE_RADIUS+5, (200, 200, 200), -1)
                image = cv2.circle(
                    image, [1150, init_pos_y+i*CIRCLE_DISTANCE-90*i+frame*3*i], CIRCLE_RADIUS, (255, 255, 255), -1)
                image = drawIcon(
                    imagelink, [1150, init_pos_y+i*CIRCLE_DISTANCE-90*i+frame*3*i], image)
            return image
        else:
            for i in range(4):

                if FingerTouch([1150, init_pos_y+i*CIRCLE_DISTANCE], finger_pos):

                    imagelink = PROJECT_FOLDER + 'facial/icons/icon_' + \
                        'EYES'+'_'+str(i)+'_'+'off.png'
                    image = cv2.circle(
                        image, [1150, init_pos_y+i*CIRCLE_DISTANCE], CIRCLE_RADIUS+5, (200, 200, 200), -1)

                    image = cv2.circle(
                        image, [1150, init_pos_y+i*CIRCLE_DISTANCE], CIRCLE_RADIUS, (255, 255, 255), -1)
                    image = drawIcon(
                        imagelink, [1150, init_pos_y+i*CIRCLE_DISTANCE], image)
                else:
                    global button_num
                    button_num = i
                    imagelink = PROJECT_FOLDER + 'facial/icons/icon_' + \
                        'EYES'+'_'+str(i)+'_'+'on.png'
                    image = cv2.circle(
                        image, [1150, init_pos_y+i*CIRCLE_DISTANCE], CIRCLE_RADIUS+5, (0, 0, 200), -1)

                    image = cv2.circle(
                        image, [1150, init_pos_y+i*CIRCLE_DISTANCE], CIRCLE_RADIUS, CIRCLE_TOUCH_COLOR, -1)
                    image = drawIcon(
                        imagelink, [1150, init_pos_y+i*CIRCLE_DISTANCE], image)
            return image
    else:
        return image


def FingerTouch(circleMid, fingerpos):
    if CIRCLE_RADIUS+FINGERTIP_RADIUS < ((circleMid[0] - fingerpos[0])**2 + (circleMid[1] - fingerpos[1])**2)**0.5:
        return True
    else:
        return False


def computePosAndResize():
    global current_menu
    if LANDMARKS != {}:
        if current_menu == 'EYES':
            pos_x = LANDMARKS[1][0]
            pos_y = LANDMARKS[18][1] - 30
            resize_x = LANDMARKS[17][0] - LANDMARKS[1][0]
            resize_y = LANDMARKS[34][1] - LANDMARKS[18][1]
            return [pos_x, pos_y, resize_x, resize_y]
        if current_menu == 'LIPS':
            pos_x = LANDMARKS[40][0]
            pos_y = LANDMARKS[51][1]
            resize_x = LANDMARKS[55][0] - LANDMARKS[49][0]
            resize_y = LANDMARKS[58][1] - LANDMARKS[51][1]
            return [pos_x, pos_y, resize_x, resize_y]
        if current_menu == 'NOSE':
            pos_x = LANDMARKS[32][0] - 15
            pos_y = LANDMARKS[28][1] + 20
            resize_x = LANDMARKS[36][0] - LANDMARKS[32][0] + 40
            resize_y = LANDMARKS[34][1] - LANDMARKS[28][1]
            return [pos_x, pos_y, resize_x, resize_y]
        return [0, 0, 0, 0]
    return [0, 0, 0, 0]


def detectnowmenu(fingerpos):
    global current_menu
    global button_num
    global fy
    if LANDMARKS != {}:
        if LANDMARKS[37][0] < fingerpos[0]+50 and LANDMARKS[40][0]+50 > fingerpos[0] and LANDMARKS[42][1]+50 > fingerpos[1] and LANDMARKS[39][1] < fingerpos[1]+50:
            if current_menu != 'EYES':
                fy = 0
                button_num = 0
            return 'EYES'
        if LANDMARKS[43][0] < fingerpos[0]+50 and LANDMARKS[46][0]+50 > fingerpos[0] and LANDMARKS[47][1]+50 > fingerpos[1] and LANDMARKS[44][1] < fingerpos[1]+50:
            if current_menu != 'EYES':
                fy = 0
                button_num = 0
            return 'EYES'
        if LANDMARKS[49][0] < fingerpos[0]+50 and LANDMARKS[55][0]+50 > fingerpos[0] and LANDMARKS[58][1]+50 > fingerpos[1] and LANDMARKS[52][1] < fingerpos[1]+50:
            if current_menu != 'LIPS':
                fy = 0
                button_num = 0
            return 'LIPS'
        if LANDMARKS[32][0] < fingerpos[0]+50 and LANDMARKS[36][0]+50 > fingerpos[0] and LANDMARKS[34][1]+50 > fingerpos[1] and LANDMARKS[28][1] < fingerpos[1]+50:
            if current_menu != 'NOSE':
                fy = 0
                button_num = 0
            return 'NOSE'
    return current_menu


cap = cv2.VideoCapture(1)
fy = 0

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('ouput.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

while True:
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.flip(image, 1)

    imgHeight = image.shape[0]
    imgWidth = image.shape[1]

    DrawFace(image)
    # Code放這下面，上面別動
    image = DrawMenu(image, fy)
    if LANDMARKS != {}:
        data = computePosAndResize()
        if data[2] != 0 and data[3] != 0:
            if button_num != 0:
                if current_menu == "EYES":
                    image = addfilter(
                        image, abs(data[0]), abs(data[1]), "/Users/EvanChen/project/facial/facial/glass" + "/glass" + str(button_num) + ".jpg", abs(data[2]), abs(data[3]))
                if current_menu == "LIPS":
                    image = addfilter(
                        image, abs(data[0]), abs(data[1]), "/Users/EvanChen/project/facial/facial/lips" + "/lip" + str(button_num) + ".jpg", abs(data[2]), abs(data[3]))
                if current_menu == "NOSE":
                    image = addfilter(
                        image, abs(data[0]), abs(data[1]), "/Users/EvanChen/project/facial/facial/nose" + "/nose" + str(button_num) + ".jpg", abs(data[2]), abs(data[3]))
    result.write(image)
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if(fy < 1000):
        fy += 2

result.release()
cv2.destroyAllWindows()
cap.release()
