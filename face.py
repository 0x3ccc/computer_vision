import cv2
import mediapipe as mp
import itertools
import matplotlib.pyplot as plt
from time import time

image = cv2.imread("56.jpg")

#mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


#landmarks

results = face_mesh.process(rgb_image)
height, width, _ = image.shape

for facial_landmarks in results.multi_face_landmarks:
        for i in range(0, 0):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)
        
            cv2.circle(image, (x, y), 5, (100, 100, 0), -1)



#shell data display
face_mesh_results = face_mesh_results = face_mesh.process(image[:,:,::-1])

LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))

if face_mesh_results.multi_face_landmarks:

        for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):

            print(f'FACE NUMBER: {face_no+1}')
            print('-----------------------')
            print(f'LEFT EYE LANDMARKS:n')
        
            for LEFT_EYE_INDEX in LEFT_EYE_INDEXES[:2]:

                print(face_landmarks.landmark[LEFT_EYE_INDEX])

            print(f'RIGHT EYE LANDMARKS:n')

            for RIGHT_EYE_INDEX in RIGHT_EYE_INDEXES[:2]:
                print(face_landmarks.landmark[RIGHT_EYE_INDEX])

#tesselate
if results.multi_face_landmarks:
        for facial_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, 
                                    landmark_list=face_landmarks,connections=mp_face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None,  
                                    connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())




cv2.imshow("image", image)

cv2.waitKey(0)
