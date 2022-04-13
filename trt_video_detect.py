import os
import time
import argparse
import cv2
import pycuda.autoinit

from tool.display_utils import open_window, set_display, show_fps, BBoxVisualization
from tool.detect_uilts import add_camera_args, Detect
from tool.utils import *
from tool.trt_yolo_plugin import Trt_yolo

# real time object detection with TensorRT optimized YOLO engine

WINDOW_NAME = 'Trt_object_detection'


def parse_args():
    desc = ('Display real-time object detection in video file'
            'with TensorRT optimized YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)

    parser.add_argument(
        '-c', '--name_file', type=str, default='data/coco.names',
        help='path of class name file')

    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.4,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-n', '--nms_thresh', type=float, default=0.6,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help='set the path of engine file ex: ./yolo.engine')
    args = parser.parse_args()
    return args

def loop_and_detect(cam, trt_yolo, conf_th, nms_th, class_list):
    # cam: the camera instance (video source)
    # trt_yolo : trt_yolo object detetor instance
    # vis : for visualization

    full_screen = False
    fps = 0.0
    tic = time.time()

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read() # 비디오로부터 프레임 반환
        if img is None:
            break

        boxes  = trt_yolo.detect(img,conf_th, nms_th) # trt engine 기반 추론
        #img = vis.draw_bboxes(img, boxes, confs,cls) # 출력 결과표시
        img = plot_boxes_cv2(img, boxes[0], class_names=class_list)
        img = show_fps(img, fps) # fps 표시
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)

        tic = toc
        key = cv2.waitKey(1)
        if key == 27: # Esc키로 quit
            break
        elif key == ord('F') or key == ord('f'):
            full_screen = not full_screen
            set_display(WINDOW_NAME, full_screen)


def main():
    args = parse_args()
    if not os.path.isfile('%s' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.engine) not found!' % args.model)

    cam = Detect(args) # video 및 카메라 인스턴스 모듈-> 출력 영상의 사이즈는 조절 가능
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # input size of network
    img_size = (416, 416)
    cls_list =load_class_names(args.name_file)

    open_window(
        WINDOW_NAME, 'TensorRT object detecion',
        cam.img_width, cam.img_height)
    # yolo 플러그인으로 detection이 구현되어있음
    trt_yolo = Trt_yolo(args.model, len(cls_list), img_size) # 이부분을 좀 잘봐야할듯
    loop_and_detect(cam, trt_yolo, args.conf_thresh, args.nms_thresh, cls_list)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()