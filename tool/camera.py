'''
IP, USB,jetson onboard 카메라 동작을 위한 class 모듈
추가적으로 video나 image input에 대한 real time detection을 지원
'''

import logging
import threading
import subprocess

import numpy as np
import cv2

# usb web cam 사용을 위한 control flag
# False라면 cv2.videocapture를 사용함 (cv2의 빌트인 모듈로 웹캠 이미지를 캡처하여 반환)
USB_GSTREAMER = True

def add_camera_args(parser):
    parser.add_argument('--image', type=str, default=None,
                        help='image file name, e.g. dog.jpg')
    parser.add_argument('--video', type=str, default=None,
                        help='video file name, e.g. traffic.mp4')
    parser.add_argument('--video_looping', action='store_true',
                        help='loop around the video file [False]')

    # 웹캠을 위한 설정, 안해도 될듯
    parser.add_argument('--rtsp', type=str, default=None,
                        help=('RTSP H.264 stream, e.g. '
                              'rtsp://admin:123456@192.168.1.64:554'))
    parser.add_argument('--rtsp_latency', type=int, default=200,
                        help='RTSP latency in ms [200]')

    # USB/ jetson 내장 카메라
    parser.add_argument('--usb', type=int, default=None,
                        help='USB webcam device id (/dev/video?) [None]')
    parser.add_argument('--gstr', type=str, default=None,
                        help='GStreamer string [None]')
    parser.add_argument('--onboard', type=int, default=None,
                        help='Jetson onboard camera [None]')
    parser.add_argument('--copy_frame', action='store_true',
                        help=('copy video frame internally [False]'))
    parser.add_argument('--do_resize', action='store_true',
                        help=('resize image/video [False]'))
    parser.add_argument('--width', type=int, default=640,
                        help='image width [640]')
    parser.add_argument('--height', type=int, default=480,
                        help='image height [480]')
    return parser


def open_cam_rtsp(uri, width, height, latency):
    """Open an RTSP URI (IP CAM)."""
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'omxh264dec' in gst_elements:
        # Use hardware H.264 decoder on Jetson platforms
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! omxh264dec ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! videoconvert ! '
                   'appsink').format(uri, latency, width, height)
    elif 'avdec_h264' in gst_elements:
        # Otherwise try to use the software decoder 'avdec_h264'
        # NOTE: in case resizing images is necessary, try adding
        #       a 'videoscale' into the pipeline
        gst_str = ('rtspsrc location={} latency={} ! '
                   'rtph264depay ! h264parse ! avdec_h264 ! '
                   'videoconvert ! appsink').format(uri, latency)
    else:
        raise RuntimeError('H.264 decoder not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    """Open a USB webcam."""
    if USB_GSTREAMER:
        gst_str = ('v4l2src device=/dev/video{} ! '
                   'video/x-raw, width=(int){}, height=(int){} ! '
                   'videoconvert ! appsink').format(dev, width, height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture(dev)


def open_cam_gstr(gstr, width, height):
    """Open camera using a GStreamer string.

    Example:
    gstr = 'v4l2src device=/dev/video0 ! video/x-raw, width=(int){width}, height=(int){height} ! videoconvert ! appsink'
    """
    gst_str = gstr.format(width=width, height=height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    """Open the Jetson onboard camera."""
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, you might need to add
        # 'flip-method=2' into gst_str below.
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    while cam.thread_running:
        _, cam.img_handle = cam.cap.read()
        if cam.img_handle is None:
            #logging.warning('Camera: cap.read() returns None...')
            break
    cam.thread_running = False


class Camera():
    """
    각 device 및 이미지/비디오 파일로 부터 이미지를 읽어들이는 클래스
    """
    def __init__(self, args):
        self.args = args
        self.is_opened = False
        self.video_file = ''
        self.video_looping = args.video_looping
        self.thread_running = False
        self.img_file = None
        self.copy_frame = args.copy_frame
        self.do_resize = args.do_resize
        self.img_width = args.width
        self.img_height = args.height
        self.cap = None
        self.thread = None
        self._open()  # try to open the camera

    def _open(self):
        if self .cpa is not None:
            raise RuntimeError('camera is already opened!')
        a = self.args

        # image detection
        if a.image:
            logging.info('detect using a image file %s' % a.image)
            self.cap = 'image'
            self.img_file = cv2.imread(a.image)
            if self.img_handle is not None:
                if self.do_resize:  # 이미지 resize
                    self.img_file = cv2.resize(self.img_file,  (a.width, a.height))
                self.is_opened = True
                self.img_height, self.img_width, _ = self.img_file.shape

        # video detection
        elif a.video:
            logging.info('detect using a video file %s' % a.video)
            self.video_file = a.video
            self.cap = cv2.VideoCapture(a.video)
            self._start()
        elif a.usb is not None:
            logging.info('Camera: using USB webcam /dev/video%d' % a.usb)
            self.cap = open_cam_usb(a.usb, a.width, a.height)
            self._start()
        elif a.gstr is not None:
            logging.info('Camera: using GStreamer string "%s"' % a.gstr)
            self.cap = open_cam_gstr(a.gstr, a.width, a.height)
            self._start()
        elif a.onboard is not None:
            logging.info('Camera: using Jetson onboard camera')
            self.cap = open_cam_onboard(a.width, a.height)
            self._start()
        else:
            raise RuntimeError('no camera type specified!')

        # elif a.rtsp:
        #     logging.info('Camera: using RTSP stream %s' % a.rtsp)
        #     self.cap = open_cam_rtsp(a.rtsp, a.width, a.height, a.rtsp_latency)
        #     self._start()

    def isOpened(self):
        return self.is_opened

    def _start(self):
        # cap 인스턴스가 닫혀있을떄
        if not self.cap.isOpened():
            logging.warning('start while cap is not opened!')
            return

        # Try to grab the 1st image and determine width and height
        _, self.img_handle = self.cap.read()
        if self.img_handle is None:
            logging.warning('Camera: cap.read() returns no image ')
            self.is_opened = False
            return

        self.is_opened = True
        # 비디오 파일을 사용할 경우
        if self.video_file:
            if not self.do_resize:
                self.img_height, self.img_width, _ = self.img_handle.shape
        else:
            self.img_height, self.img_width, _ = self.img_handle.shape
            # 만약 video file 소스가 따로 없고, 캠을 사용할 경우, child thread를 시작
            assert not self.thread_running
            self.thread_running = True
            self.thread = threading.Thread(target=graph_img, args=(self,))
            self.thread.start()

    def _stop(self):
        if self.thread_running:
            self.thread_running = False
            # self.thread. join

    def read(self):
        # camera object로 부터 프레임을 읽어들임
        if not self.is_opened:
            return None
        if self.video_file:
            _, img = self.cap.read()
            if img is None:
                logging.infor('Camera: reach to end of the video')
                if self.video_looping:


        elif self.cap == 'image':
            pass
        else:
            if self.copy_frame:
                r