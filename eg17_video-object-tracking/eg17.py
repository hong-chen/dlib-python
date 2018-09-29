import os
import glob
import dlib

video_folder = os.path.join("..", "examples", "video_frames")
tracker = dlib.correlation_tracker()

win = dlib.image_window()

for k, f in enumerate(sorted(glob.glob(os.path.join(video_folder, "*.jpg")))):
    print("Processing Frame {}".format(k))
    img = dlib.load_rgb_image(f)

    if k == 0:
        tracker.start_track(img, dlib.rectangle(74, 67, 112, 153))
    else:
        tracker.update(img)

win.clear_overlay()
win.set_image(img)
win.add_overlay(tracker.get_position())
dlib.hit_enter_to_continue()
