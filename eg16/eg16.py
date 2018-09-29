import os
import sys
import glob
import dlib

face_folder = sys.argv[1]
options = dlib.shape_predictor_training_options()

options.oversampling_amount = 300
options.nu = 0.05
options.tree_depth = 2
options.be_verbose = True

training_xml_path = os.path.join(face_folder, "training_with_face_landmarks.xml")
dlib.train_shape_predictor(training_xml_path, "predictor.dat", options)
