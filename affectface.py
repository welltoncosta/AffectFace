from imutils import face_utils
import datetime
import imutils
import time
import dlib
import cv2, math
import numpy as np
from imutils import face_utils, rotate_bound
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

from emonet.models import EmoNet

import mariadb
import sys

#### MUITO IMPORTANTE MUDAR PARA CADA USUARIO #####
id_usuario=1;
###################################################

conn = mariadb.connect(
        user="root",
        password="123",
        host="localhost",
        port=3306,
        database="doutorado"
)
print(conn)
cur = conn.cursor()

# Usar número de experssão 5 ou 8?
n_expression=5

net = EmoNet(n_expression=n_expression).to()

# Carregando o modelo do arquivo PTH
state_dict_path = Path(__file__).parent.joinpath('.', f'emonet_{n_expression}.pth')

print(f'Carregando o modelo de {state_dict_path}.')

state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}

net.load_state_dict(state_dict, strict=False)
net.eval()

transform_image = transforms.Compose([transforms.ToTensor()])

print("[INFO] loading facial landmark predictor...")
model = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model) # link to model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


video_capture = cv2.VideoCapture(0)
#video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#video_capture = cv2.imread(sys.argv[1])
cv2.imshow('Video', np.empty((5,5),dtype=float))

#points are tuples in the form (x,y)
# returns angle between points in degrees
def calculate_inclination(point1, point2):
    x1,x2,y1,y2 = point1[0], point2[0], point1[1], point2[1]
    incl = -180/math.pi*math.atan((float(y2-y1))/(x2-x1))
    return incl



def calculate_boundbox(list_coordinates):
    x = min(list_coordinates[:,0])
    y = min(list_coordinates[:,1])
    w = max(list_coordinates[:,0]) - x
    h = max(list_coordinates[:,1]) - y
    return (x,y,w,h)

def get_face_boundbox(points, face_part):
    if face_part == 1:
        (x,y,w,h) = calculate_boundbox(points[17:22]) #left eyebrow
    elif face_part == 2:
        (x,y,w,h) = calculate_boundbox(points[22:27]) #right eyebrow
    elif face_part == 3:
        (x,y,w,h) = calculate_boundbox(points[36:42]) #left eye
    elif face_part == 4:
        (x,y,w,h) = calculate_boundbox(points[42:48]) #right eye
    elif face_part == 5:
        (x,y,w,h) = calculate_boundbox(points[29:36]) #nose
    elif face_part == 6:
        (x,y,w,h) = calculate_boundbox(points[48:68]) #mouth
    return (x,y,w,h)


#função para atualizar o gráfico
def _update_plot(i, fig, scat, valence, arousal):
 print("valence = ", valence, "\narousal = ", arousal)
 x = valence
 y = arousal

 scat.set_offsets([x,y])
 return scat

while cv2.getWindowProperty('Video', 0) >= 0:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detecta faces no frame em nível de cinza
    rects = detector(gray, 0)

    # loop sobre detecção de faces
    for rect in rects:

        # determina os pontos de referência faciais para a região do rosto e, em seguida,
        # converte as coordenadas do ponto de referência facial (x, y) em uma matriz NumPy
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        incl = calculate_inclination(shape[17], shape[26])

        #print ("Pixels distance points in mouth: ", shape[66][1] - shape[62][1])
        x,y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        if y>0 and x>0:
            crop_img = frame[y:y+h, x:x+w]
            dim = (256, 256)

            resized = cv2.resize(crop_img, dim)

            image_cropada = transform_image(resized)
            image_cropada = image_cropada.unsqueeze(0)
            #print(image_cropada)
            with torch.no_grad():
                out = net(image_cropada)

            expr = out['expression']
            expr = np.argmax(np.squeeze(expr.cpu().numpy()), axis=0)

            val = out['valence']
            ar = out['arousal']

            val = np.squeeze(val.cpu().numpy())
            ar = np.squeeze(ar.cpu().numpy())

            valence_pred = val
            arousal_pred = ar
            expression_pred = expr

            print("valence = ", valence_pred, "\narousal = ", arousal_pred, "\n\n")
            print("expressao = ", expression_pred, "\n\n")

            valence = float(valence_pred)
            arousal = float(arousal_pred)

            sql = "INSERT INTO face_experimento (id_usuario, expressao, valence, arousal, horario) VALUES (%s, %s, %s, %s, NOW())";
            val = (id_usuario, expression_pred, valence, arousal)
            cur.execute(sql, val);

            print("inserido")
            conn.commit()

            #cv2.imshow('./cropped', crop_img)

        # faz um loop sobre as coordenadas (x, y) para os pontos de referência
        # faciais e desenhe-os na imagem

        ti=0
        branco = (255, 255, 255)
        verde = (0, 255, 0)
        azul = (255, 0, 0)
        vermelho = (0, 0, 255)

        for (x, y) in shape:

            if ti==0:
                point1 = (x, y)

            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

            point2 = (x, y)

            if ti!=17 and ti!=22 and ti!=27 and ti!=31 and ti!=36 and ti!=48 and ti!=42 and ti!=60:
             cv2.line(frame, point1, point2, branco, 2)

            if ti!=17 and ti!=22 and ti!=27 and ti!=31 and ti!=36 and ti!=48 and ti!=42 and ti!=60:
             cv2.line(frame, point1, point2, branco, 2)


            if ti+1==1:
                    ponto1 = point2
            if ti+1==18:
                    cv2.line(frame, ponto1, point2, branco, 1)

            if ti+1==17:
                    ponto17 = point2
            if ti+1==27:
                    cv2.line(frame, ponto17, point2, branco, 1)

            if ti+1==9:
                    ponto9 = point2
            if ti+1==58:
                    cv2.line(frame, ponto9, point2, branco, 1)

            if ti+1==21:
                    ponto21 = point2
            if ti+1==24:
                    cv2.line(frame, ponto21, point2, branco, 1)
            if ti+1==57:
                    cv2.line(frame, ponto9, point2, branco, 1)

            if ti+1==30:
                    ponto30 = point2
            if ti+1==32:
                    cv2.line(frame, ponto30, point2, branco, 1)
            if ti+1==36:
                    cv2.line(frame, ponto30, point2, branco, 1)

            if ti+1==37:
                point37 = point2
            if ti+1==43:
                cv2.line(frame, point1, point37, branco, 2)
            if ti+1==43:
                point43 = point2
            if ti+1==49:
             cv2.line(frame, point1, point43, branco, 2)

            if ti+1==49:
                point49 = point2
            if ti+1==61:
             cv2.line(frame, point1, point49, branco, 2)

            if ti+1==61:
                point61 = point2
            if ti+1==68:
             cv2.line(frame, point61, point2, branco, 2)



            if ti+1==7:
                    ponto7 = point2
            if ti+1==49:
                    cv2.line(frame, ponto7, point2, branco, 1)
            if ti+1==60:
                    cv2.line(frame, ponto7, point2, branco, 1)

            if ti+1==8:
                    ponto8 = point2
            if ti+1==59:
                    cv2.line(frame, ponto8, point2, branco, 1)
            if ti+1==60:
                    cv2.line(frame, ponto8, point2, branco, 1)

            if ti+1==9:
                    ponto9 = point2
            if ti+1==57:
                    cv2.line(frame, ponto9, point2, branco, 1)
            if ti+1==58:
                    cv2.line(frame, ponto9, point2, branco, 1)
            if ti+1==59:
                    cv2.line(frame, ponto9, point2, branco, 1)

            if ti+1==1:
                    ponto1 = point2
            if ti+1==37:
                    cv2.line(frame, ponto1, point2, branco, 1)

            if ti+1==2:
                    ponto2 = point2
            if ti+1==32:
                    cv2.line(frame, ponto2, point2, branco, 1)
            if ti+1==37:
                    cv2.line(frame, ponto2, point2, branco, 1)
            if ti+1==42:
                    cv2.line(frame, ponto2, point2, branco, 1)

            if ti+1==17:
                    ponto17 = point2
            if ti+1==46:
                    cv2.line(frame, ponto17, point2, branco, 1)

            if ti+1==16:
                    ponto16 = point2
            if ti+1==36:
                    cv2.line(frame, ponto16, point2, branco, 1)
            if ti+1==46:
                    cv2.line(frame, ponto16, point2, branco, 1)
            if ti+1==47:
                    cv2.line(frame, ponto16, point2, branco, 1)

            if ti+1==2:
                    ponto2 = point2
            if ti+1==37:
                    cv2.line(frame, ponto2, point2, branco, 1)
            if ti+1==42:
                    cv2.line(frame, ponto2, point2, branco, 1)


            if ti+1==10:
                    ponto10 = point2
            if ti+1==57:
                    cv2.line(frame, ponto10, point2, branco, 1)
            if ti+1==56:
                    cv2.line(frame, ponto10, point2, branco, 1)

            if ti+1==11:
                    ponto11 = point2
            if ti+1==55:
                    cv2.line(frame, ponto11, point2, branco, 1)
            if ti+1==56:
                    cv2.line(frame, ponto11, point2, branco, 1)

            if ti+1==12:
                    ponto12 = point2
            if ti+1==55:
                    cv2.line(frame, ponto12, point2, branco, 1)
            if ti+1==13:
                    ponto13 = point2
            if ti+1==55:
                    cv2.line(frame, ponto13, point2, branco, 1)
            if ti+1==14:
                    ponto14 = point2
            if ti+1==55:
                    cv2.line(frame, ponto14, point2, branco, 1)
            if ti+1==15:
                    ponto15 = point2
            if ti+1==55:
                    cv2.line(frame, ponto15, point2, branco, 1)

            if ti+1==3:
                    ponto3 = point2
            if ti+1==49:
                    cv2.line(frame, ponto3, point2, branco, 1)
            if ti+1==4:
                    ponto4 = point2
            if ti+1==49:
                    cv2.line(frame, ponto4, point2, branco, 1)
            if ti+1==5:
                    ponto5 = point2
            if ti+1==49:
                    cv2.line(frame, ponto5, point2, branco, 1)
            if ti+1==6:
                    ponto6 = point2
            if ti+1==49:
                    cv2.line(frame, ponto6, point2, branco, 1)
            if ti+1==42:
                    ponto42 = point2
            if ti+1==49:
                    cv2.line(frame, ponto42, point2, branco, 1)

            if ti+1==32:
                    ponto32 = point2
            if ti+1==51:
                    cv2.line(frame, ponto32, point2, branco, 1)

            if ti+1==34:
                    ponto34 = point2
            if ti+1==51:
                    cv2.line(frame, ponto34, point2, branco, 1)
            if ti+1==52:
                    cv2.line(frame, ponto34, point2, branco, 1)
            if ti+1==53:
                    cv2.line(frame, ponto34, point2, branco, 1)

            if ti+1==31:
                    ponto31 = point2
            if ti+1==32:
                    cv2.line(frame, ponto31, point2, branco, 1)
            if ti+1==33:
                    cv2.line(frame, ponto31, point2, branco, 1)
            if ti+1==34:
                    cv2.line(frame, ponto31, point2, branco, 1)
            if ti+1==35:
                    cv2.line(frame, ponto31, point2, branco, 1)
            if ti+1==36:
                    cv2.line(frame, ponto31, point2, branco, 1)

            if ti+1==3:
                    ponto3 = point2
            if ti+1==32:
                    cv2.line(frame, ponto3, point2, branco, 1)

            if ti+1==15:
                    ponto15 = point2
            if ti+1==36:
                    cv2.line(frame, ponto15, point2, branco, 1)

            if ti+1==32:
                    ponto32 = point2
            if ti+1==49:
                    cv2.line(frame, ponto32, point2, branco, 1)
            if ti+1==50:
                    cv2.line(frame, ponto32, point2, branco, 1)
            if ti+1==61:
                    cv2.line(frame, ponto32, point2, branco, 1)

            if ti+1==36:
                    ponto36 = point2
            if ti+1==53:
                    cv2.line(frame, ponto36, point2, branco, 1)
            if ti+1==65:
                    cv2.line(frame, ponto36, point2, branco, 1)
            if ti+1==54:
                    cv2.line(frame, ponto36, point2, branco, 1)
            if ti+1==54:
                    cv2.line(frame, ponto36, point2, branco, 1)
            if ti+1==55:
                    cv2.line(frame, ponto36, point2, branco, 1)
            if ti+1==48:
                    cv2.line(frame, ponto36, point2, branco, 1)
            if ti+1==47:
                    cv2.line(frame, ponto36, point2, branco, 1)


            if ti+1==32:
                    ponto32 = point2
            if ti+1==41:
                    cv2.line(frame, ponto32, point2, branco, 1)
            if ti+1==42:
                    cv2.line(frame, ponto32, point2, branco, 1)

            if ti+1==47:
                    ponto47 = point2
            if ti+1==55:
                    cv2.line(frame, ponto47, point2, branco, 1)

            if ti+1==22:
                    ponto22 = point2
            if ti+1==39:
                    cv2.line(frame, ponto22, point2, branco, 1)
            if ti+1==40:
                    cv2.line(frame, ponto22, point2, branco, 1)
            if ti+1==28:
                    cv2.line(frame, ponto22, point2, branco, 1)
            if ti+1==23:
                    cv2.line(frame, ponto22, point2, branco, 1)
            if ti+1==24:
                    cv2.line(frame, ponto22, point2, branco, 1)

            if ti+1==23:
                    ponto23 = point2
            if ti+1==28:
                    cv2.line(frame, ponto23, point2, branco, 1)
            if ti+1==43:
                    cv2.line(frame, ponto23, point2, branco, 1)
            if ti+1==44:
                    cv2.line(frame, ponto23, point2, branco, 1)

            if ti+1==30:
                    ponto30 = point2
            if ti+1==40:
                    cv2.line(frame, ponto30, point2, branco, 1)
            if ti+1==41:
                    cv2.line(frame, ponto30, point2, branco, 1)
            if ti+1==43:
                    cv2.line(frame, ponto30, point2, branco, 1)
            if ti+1==48:
                    cv2.line(frame, ponto30, point2, branco, 1)

            if ti+1==31:
                    ponto31 = point2
            if ti+1==41:
                    cv2.line(frame, ponto31, point2, branco, 1)
            if ti+1==48:
                    cv2.line(frame, ponto31, point2, branco, 1)

            if ti+1==28:
                    ponto28 = point2
            if ti+1==40:
                    cv2.line(frame, ponto28, point2, branco, 1)
            if ti+1==43:
                    cv2.line(frame, ponto28, point2, branco, 1)

            if ti+1==18:
                    ponto18 = point2
            if ti+1==37:
                    cv2.line(frame, ponto18, point2, branco, 1)

            if ti+1==19:
                    ponto19 = point2
            if ti+1==37:
                    cv2.line(frame, ponto19, point2, branco, 1)
            if ti+1==38:
                    cv2.line(frame, ponto19, point2, branco, 1)

            if ti+1==20:
                    ponto20 = point2
            if ti+1==38:
                    cv2.line(frame, ponto20, point2, branco, 1)
            if ti+1==39:
                    cv2.line(frame, ponto20, point2, branco, 1)

            if ti+1==21:
                    ponto21 = point2
            if ti+1==39:
                    cv2.line(frame, ponto21, point2, branco, 1)

            if ti+1==22:
                    ponto22 = point2
            if ti+1==39:
                    cv2.line(frame, ponto22, point2, branco, 1)
            if ti+1==40:
                    cv2.line(frame, ponto22, point2, branco, 1)

            if ti+1==27:
                    ponto27 = point2
            if ti+1==46:
                    cv2.line(frame, ponto27, point2, branco, 1)

            if ti+1==26:
                    ponto26 = point2
            if ti+1==45:
                    cv2.line(frame, ponto26, point2, branco, 1)
            if ti+1==46:
                    cv2.line(frame, ponto26, point2, branco, 1)

            if ti+1==25:
                    ponto25 = point2
            if ti+1==44:
                    cv2.line(frame, ponto25, point2, branco, 1)
            if ti+1==45:
                    cv2.line(frame, ponto25, point2, branco, 1)

            if ti+1==24:
                    ponto24 = point2
            if ti+1==44:
                    cv2.line(frame, ponto24, point2, branco, 1)

            if ti+1==23:
                    ponto23 = point2
            if ti+1==43:
                    cv2.line(frame, ponto23, point2, branco, 1)
            if ti+1==44:
                    cv2.line(frame, ponto23, point2, branco, 1)

            if ti+1==38:
                    ponto38 = point2
            if ti+1==42:
                    cv2.line(frame, ponto38, point2, branco, 1)

            if ti+1==39:
                    ponto39 = point2
            if ti+1==41:
                    cv2.line(frame, ponto39, point2, branco, 1)
            if ti+1==42:
                    cv2.line(frame, ponto39, point2, branco, 1)

            if ti+1==45:
                    ponto45 = point2
            if ti+1==47:
                    cv2.line(frame, ponto45, point2, branco, 1)

            if ti+1==44:
                    ponto44 = point2
            if ti+1==47:
                    cv2.line(frame, ponto44, point2, branco, 1)
            if ti+1==48:
                    cv2.line(frame, ponto44, point2, branco, 1)

            if ti+1==52:
                    ponto52 = point2
            if ti+1==62:
                    cv2.line(frame, ponto52, point2, branco, 1)
            if ti+1==63:
                    cv2.line(frame, ponto52, point2, branco, 1)
            if ti+1==64:
                    cv2.line(frame, ponto52, point2, branco, 1)

            if ti+1==60:
                    ponto60 = point2
            if ti+1==61:
                    cv2.line(frame, ponto60, point2, branco, 1)
            if ti+1==67:
                    cv2.line(frame, ponto60, point2, branco, 1)
            if ti+1==68:
                    cv2.line(frame, ponto60, point2, branco, 1)

            if ti+1==56:
                    ponto56 = point2
            if ti+1==65:
                    cv2.line(frame, ponto56, point2, branco, 1)
            if ti+1==66:
                    cv2.line(frame, ponto56, point2, branco, 1)
            if ti+1==67:
                    cv2.line(frame, ponto56, point2, branco, 1)

            if ti+1==59:
                    ponto59 = point2
            if ti+1==67:
                    cv2.line(frame, ponto59, point2, branco, 1)
            if ti+1==58:
                    ponto58 = point2
            if ti+1==67:
                    cv2.line(frame, ponto58, point2, branco, 1)
            if ti+1==57:
                    ponto57 = point2
            if ti+1==67:
                    cv2.line(frame, ponto57, point2, branco, 1)

            ti = ti + 1

            point1 = point2

    cv2.imshow('./Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
    	break

video_capture.release()
cv2.destroyAllWindows()
