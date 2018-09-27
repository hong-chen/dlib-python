import sys
import dlib

# CNN means Convolutional Neural Network

cnn_face_detector = dlib.cnn_face_detection_model_v1(sys.argv[1])
win = dlib.image_window()

for f in sys.argv[2:]:
    print('Processing file: {}'.format(f))
    img = dlib.load_rgb_image(f)
    dets = cnn_face_detector(img, 1)
    print('Number of faces detected: {}'.format(len(dets)))
    for i, d in enumerate(dets):
        print('Detection: {} Left: {} Top: {} Right: {} Bottom: {} Confidence: {}'.format(
                i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

    rects = dlib.rectangles()
    rects.extend([d.rect for d in dets])

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(rects)
    dlib.hit_enter_to_continue()
