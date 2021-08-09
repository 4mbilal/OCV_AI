from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import socket as Socket
from io import BytesIO
import depthai as dai
from PIL import Image
import numpy as np
import threading
import argparse
import queue
import cv2

q = queue.Queue(maxsize=1)

# Create pipeline
pipeline = dai.Pipeline()
queueNames = []

# Define sources and outputs
camRgb = pipeline.createColorCamera()
left = pipeline.createMonoCamera()
right = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

rgbOut = pipeline.createXLinkOut()
depthOut = pipeline.createXLinkOut()

rgbOut.setStreamName("rgb")
queueNames.append("rgb")
depthOut.setStreamName("depth")
queueNames.append("depth")

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setIspScale(2, 3)
camRgb.setPreviewSize(300, 300)
# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
camRgb.initialControl.setManualFocus(130)

left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setConfidenceThreshold(200)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
# Linking
camRgb.isp.link(rgbOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.disparity.link(depthOut.input)


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        global q
        print('connection from:', self.address_string())

        if self.path == '/':
            self.send_response(200)
            self.send_header(
                'Content-type',
                'multipart/x-mixed-replace; boundary=--jpgboundary'
            )
            self.end_headers()

            while True:
                frame = q.get(block=True)
                self.wfile.write(b'----jpgboundary\r\n')
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', len(frame))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
        else:
            print('error', self.path)
            self.send_response(404)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>')
            self.wfile.write('<h1>{0!s} not found</h1>'.format(self.path))
            self.wfile.write('</body></html>')


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


def get_stream(q,opCode):
    with dai.Device(pipeline) as device:

        qrgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        qdepth = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

        frameRgb = None
        frameDepth = None
        while True:
            frameRgb = qrgb.get().getCvFrame()
            frameDepth = qdepth.get().getCvFrame()
            frameDepth = cv2.medianBlur(frameDepth, 15)
            frameDepth = frameDepth / np.max(frameDepth)
            frameDepth = 1 - frameDepth

            if frameRgb is not None and frameDepth is not None:
                frameRgb = frameRgb.astype(np.float32)
                frameRgb = cv2.cvtColor(frameRgb, cv2.COLOR_BGR2RGB)

                if opCode == 0:
                    frameRgb = frameRgb.astype(np.float32)
                    frameRgb = frameRgb / 255.0
                    # BGR into HLS color space conversion
                    framehls = cv2.cvtColor(frameRgb, cv2.COLOR_RGB2HLS)
                    framehls[:, :, 2] = (1 - frameDepth) * framehls[:, :, 2]
                    img = cv2.cvtColor(framehls, cv2.COLOR_HLS2RGB) * 255
                    img = img[:, (img.shape[1] // 4):(3 * img.shape[1] // 4)]
                elif opCode == 1:
                    frameBlur = cv2.GaussianBlur(frameRgb, (51, 51), 0)
                    img = frameRgb.copy()
                    img[:, :, 0] = (frameDepth * frameBlur[:, :, 0]) + ((1 - frameDepth) * frameRgb[:, :, 0])
                    img[:, :, 1] = (frameDepth * frameBlur[:, :, 1]) + ((1 - frameDepth) * frameRgb[:, :, 1])
                    img[:, :, 2] = (frameDepth * frameBlur[:, :, 2]) + ((1 - frameDepth) * frameRgb[:, :, 2])
                    img = img[:, (img.shape[1] // 4):(3 * img.shape[1] // 4)]
                else:
                    img = cv2.resize(frameRgb, (frameRgb.shape[0], frameRgb.shape[1]//2))
                res = np.concatenate((img, img), axis=1)
                buf = BytesIO()
                frame = Image.fromarray(res.astype('uint8'))
                frame.save(buf, format='JPEG')
                jpeg_img = buf.getvalue()
                q.put(jpeg_img)

def handleArgs():
    parser = argparse.ArgumentParser(description='A simple program to help people with monocular vision: -a 192.168.1.1 -p 8080 -o 3')
    parser.add_argument('-p', '--port', help='HTTP server port, default is 8080', type=int, default=8080)
    parser.add_argument('-o', '--opcode', help='operation code mode 0: Saturation 1:Blurring 2:Extended view, default is 0', type=int, default=0)
    parser.add_argument('-a', '--address', help='the ip address, default is localhost', default='localhost')

    args = vars(parser.parse_args())
    return args

args = handleArgs()

t1 = threading.Thread(target=get_stream, args=(q,args["opcode"]))
t1.start()
server = ThreadedHTTPServer((args["address"], args["port"]), Handler)
server.serve_forever()

















