import json
import time
import argparse
import cv2
import numpy as np
import mvnc.mvncapi as mvncapi
import mvnc.mvncapi as mvncapi

def get_mvnc_device():
    mvncapi.global_set_option(mvncapi.GlobalOption.RW_LOG_LEVEL, 0)
    devices = mvncapi.enumerate_devices()
    if (len(devices) < 1):
        print("Error - no NCS devices detected")
        quit()
    dev = mvncapi.Device(devices[0])
    try:
        dev.open()
    except:
        print("Error - Could not open NCS device.")
        quit()
    return dev

def load_graph(dev, graph_file):
    with open(graph_file, mode='rb') as f:
        graph_file_buffer = f.read()
    graph = mvncapi.Graph('graph1')
    input_fifo, output_fifo = graph.allocate_with_fifos(dev, graph_file_buffer)
    return graph, input_fifo, output_fifo

import cv2
import numpy as np
import json
from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor

def get_meta(meta_file):
    with open(meta_file, 'r') as fp:
        meta = json.load(fp)
    return meta

def pre_proc_img(im, meta):
    w, h, _ = meta['inp_size']
    im = im.astype(np.float32)
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1]
    imsz = np.expand_dims(imsz, axis=0)
    return imsz

def findboxes_meta(meta, net_out):
    boxes = list()
    boxes = box_constructor(meta, net_out)
    return boxes

def process_box_meta(meta, b, h, w, threshold):
    max_indx = np.argmax(b.probs)
    max_prob = b.probs[max_indx]
    label = meta['labels'][max_indx]
    if max_prob > threshold:
        left = int((b.x - b.w/2.) * w)
        right = int((b.x + b.w/2.) * w)
        top = int((b.y - b.h/2.) * h)
        bot = int((b.y + b.h/2.) * h)
        if left < 0:
            left = 0
        if right > w - 1:
            right = w - 1
        if top < 0:
            top = 0
        if bot > h - 1:
            bot = h - 1
        mess = '{}'.format(label)
        return (left, right, top, bot, mess, max_indx, max_prob)
    return None

def procces_out(out, meta, img_orig_dimensions):
    h, w, _ = img_orig_dimensions
    boxes = findboxes_meta(meta, out)
    threshold = meta['thresh']
    boxesInfo = list()
    for box in boxes:
        tmpBox = process_box_meta(meta, box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

def add_bb_to_img(img_orig, boxes):
    for box in boxes:
        left = box["topleft"]['x']
        right = box["bottomright"]['x']
        top = box["topleft"]['y']
        bot = box["bottomright"]['y']
        mess = box["label"]
        confidence = box["confidence"]
        h, w, _ = img_orig.shape
        thick = int((h + w) // 300)
        cv2.rectangle(img_orig,
                      (left, top), (right, bot),
                      [255, 0, 0], 2)
        cv2.putText(
            img_orig, mess, (left, top - 12),
            0, 1e-3 * h, [255, 0, 0],
            2)


def inference_video(graph_file,
                    meta_file,
                    video_in_name,
                    video_out_name,
                    threshold,
                    show):
    meta = get_meta(meta_file)
    meta['thresh'] = threshold
    dev = get_mvnc_device()
    graph, input_fifo, output_fifo = load_graph(dev, graph_file)
    cap = cv2.VideoCapture()
    cap.open(video_in_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_out_name, fourcc, fps, (width,height))
    times = []
    winName = 'Deep learning object detection'
    if show:
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winName, (800, 600))
    # cap.read(300)
    i = 0
    while cv2.waitKey(1) < 0:
        for j in [0,20]:
            ret, frame = cap.read()

        if not ret:
            print("Video Ended")
            break
        frame_orig = np.copy(frame)
        img_orig_dimensions = frame_orig.shape
        frame = pre_proc_img(frame, meta)
        start = time.time()
        graph.queue_inference_with_fifo_elem(
            input_fifo, output_fifo, frame, 'user object')
        output, _ = output_fifo.read_elem()
        end = time.time()
        print('FPS: {:.2f}'.format((1 / (end - start))))
        times.append((1/ (end - start)))
        y_out = np.reshape(output, (13, 13, 425))
        y_out = np.squeeze(y_out)
        boxes = procces_out(y_out, meta, img_orig_dimensions)
        add_bb_to_img(frame_orig, boxes)
        out.write(frame_orig)
        if show:
            cv2.imshow(winName, frame_orig)
        i += 1
    cap.release()
    out.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--input_video",
        required=True,
        help="path to input video")
    ap.add_argument(
        "-o", "--output_video",
        required=False,
        default='out.avi',
        help="path to output video")
    ap.add_argument(
        "-m", "--meta_file",
        required=False,
        default='tiny-yolo-voc.meta',
        help="path to meta file")
    ap.add_argument(
        "-mg", "--movidius_graph",
        required=False,
        default= 'tiny-yolo-voc.graph',
        help="path to movidius graph")
    ap.add_argument(
        "-th", "--threshold",
        required=False,
        default = 0.3,
        help="threshold")
    ap.add_argument(
        "-s", "--show",
        required=False,
        default=False,
        help="show it on the screen")
    args = ap.parse_args()
    inference_video(
        args.movidius_graph,
        args.meta_file,
        args.input_video,
        args.output_video,
        args.threshold,
        args.show)

if __name__ == '__main__':
    main()

'''
python3 mymovidius_yolo_test.py \
    -i "bike_lanes.mp4" \
    -m "yolov2-tiny.meta" \
    -mg "yolov2-tiny.graph" 
'''