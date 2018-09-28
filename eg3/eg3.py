import sys
import dlib

predictor_path = sys.argv[1]
face_file_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

img = dlib.load_rgb_image(face_file_path)

dets = detector(img, 1)
num_faces = len(dets)

if num_faces == 0:
    print("Sorry, there were no faces found in '{}'".format(face_file_path))
    exit()

faces = dlib.full_object_detections()
for detection in dets:
    faces.append(sp(img, detection))

window = dlib.image_window()
images = dlib.get_face_chips(img, faces, size=320)
for image in images:
    window.set_image(image)
    dlib.hit_enter_to_continue()

image = dlib.get_face_chip(img, faces[0])
window.set_image(image)
dlib.hit_enter_to_continue()
