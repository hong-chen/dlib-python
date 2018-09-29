import sys
import os
import dlib
import glob

predictor_path      = sys.argv[1]
face_rec_model_path = sys.argv[2]
face_folder_path    = sys.argv[3]
output_folder_path  = sys.argv[4]

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

descriptors = []
images = []

for f in glob.glob(os.path.join(face_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        shape = sp(img, d)

        face_descriptor = facerec.compute_face_descriptor(img, shape)
        descriptors.append(face_descriptor)
        images.append((img, shape))

labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
num_classes = len(set(labels))
print("Number of clusters: {}".format(num_classes))

biggest_class = None
biggest_class_length = 0
for i in range(0, num_classes):
    class_length = len([label for label in labels if label==i])
    if class_length > biggest_class_length:
        biggest_class_length = class_length
        biggest_class = i

print("Biggest cluster id number: {}".format(biggest_class))
print("Number of faces in biggest cluster: {}".format(biggest_class_length))

indices = []
for i, label in enumerate(labels):
    if label == biggest_class:
        indices.append(i)

print("Indices of images in the biggest cluster: {}".format(str(indices)))

if not os.path.isdir(output_folder_path):
    os.makedirs(output_folder_path)

print("Saving faces in largest cluster to output folder...")
for i, index in enumerate(indices):
    img, shape = images[index]
    file_path = os.path.join(output_folder_path, "faces_"+str(i))
    dlib.save_face_chip(img, shape, file_path, size=150, padding=0.25)
