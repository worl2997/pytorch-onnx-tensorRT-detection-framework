# real time object detection with TensorRT optimized YOLO engine

import os
import time
import argparse
import cv2
import pycuda.autoinit

from tool.utils import load_class_names
from tool.display_utils import open_window, set_display, show_fps, BBoxVisualization
from tool.camera import add_camera_args, Camera
WINDOW_NAME = 'Trt_object_detection'


def parse_args():
    desc = ('Display real-time object detection in video file'
            'with TensorRT optimized YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)

    parser.add_argument(
        '-c', '--name_file', type=str, default='coco.names',
        help='path of class name file')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    args = parser.parse_args()
    return args

def loop_and_detect(cam, trt_yolo, conf_th, vis):
    # cam: the camera instance (video source)
    # trt_yolo : trt_yolo object detetor instance
    # vis : for visualization

    full_screen = False
    fps = 0.0
    tic = time.time()

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read() # 이미지를 반환
        if img is None:
            break

        boxes, confs, cls = trt_yolo.detect(img,conf_th) # trt engine 기반 추론
        img = vis.draw_bboxes(img, boxes, confs,cls) # 출력 결과표시
        img = show_fps(img,fps) # fps 표시
        cv2.imshow(WINDOW_NAME, img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27: # Esc키로 quit
            break
        elif key ==ord('F') or key == ord('f'):
            full_screen = not full_screen
            set_display(WINDOW_NAME, full_screen)


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args) # video 및 카메라 인스턴스 모듈
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls = load_class_names(args.name_file)
    vis = BBoxVisualization(cls)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box) # 이부분을 좀 잘봐야할듯

    open_window(
        WINDOW_NAME, 'TensorRT object detecion',
        cam.img_width, cam.img_height)

    loop_and_detect(cam,trt_yolo, args.conf_thresh,vis=vis)

    cam.releas()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()