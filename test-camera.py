import cv2
import numpy as np
import sys, getopt, time

try:
    SHOW_IMAGE = True
    import tensorflow.lite as tflite
except ImportError:
    SHOW_IMAGE = False
    import tflite_runtime.interpreter as tflite
    

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240


def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path) as f:
        labels = {}
        index = 0
        for line in f.readlines():
            labels[index] = line.rstrip("\n")
            index = index + 1
        return labels


def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index):
    r"""Process an image, Return a list of detected class ids and positions"""
    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim

    # Process
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get outputs
    output_details = interpreter.get_output_details()
    # output_details[0] - position
    # output_details[1] - class id
    # output_details[2] - score
    # output_details[3] - count

    positions = np.squeeze(interpreter.get_tensor(output_details[1]["index"]))
    classes = np.squeeze(interpreter.get_tensor(output_details[3]["index"]))
    scores = np.squeeze(interpreter.get_tensor(output_details[0]["index"]))

    result = []

    for idx, score in enumerate(scores):
        if score > 0.2:
            result.append({"pos": positions[idx], "_id": classes[idx], "score": score})

    return result


def display_result(result, frame, labels):
    r"""Display Detected Objects"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.6
    color = (255, 0, 0)  # Blue color
    thickness = 1

    # let's resize our image to be 150 pixels wide, but in order to
    # prevent our resized image from being skewed/distorted, we must
    # first calculate the ratio of the *new* width to the *old* width
    r = 640.0 / frame.shape[1]
    dim = (640, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # position = [ymin, xmin, ymax, xmax]
    # x * IMAGE_WIDTH
    # y * IMAGE_HEIGHT
    width = frame.shape[1]
    height = frame.shape[0]

    for obj in result:
        pos = obj["pos"]
        _id = obj["_id"]
        score = obj["score"] * 100

        x1 = int(pos[1] * width)
        x2 = int(pos[3] * width)
        y1 = int(pos[0] * height)
        y2 = int(pos[2] * height)

        cv2.putText(frame, labels[_id], (x1, y1), font, size, color, thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow("Object Detection", frame)


def main(argv):

    dir = None
    
    try:
        
        opts, _ = getopt.getopt(argv, "hd:", ["help","directory="])
        for opt, arg in opts:
            if opt == "-h, --help":
                raise Exception()
            elif opt in ("-d", "--directory"):
                dir = str(arg)

        if (dir is None ):
            raise Exception()

    except Exception:
        print("Specify a directory in which the model is located.")
        print("test-camera.py -d <directory>")
        sys.exit(2)


    model_path = dir + "/model.tflite"
    label_path = dir + "/labels.txt"

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)

    interpreter = load_model(model_path)
    labels = load_labels(label_path)

    input_details = interpreter.get_input_details()

    # Get Width and Height
    input_shape = input_details[0]["shape"]
    height = input_shape[1]
    width = input_shape[2]

    # Get input index
    input_index = input_details[0]["index"]

    # Process Stream
    while True:
        
        ret, frame = cap.read()

        if ret:

            start = round(time.time() * 1000)
            image = cv2.resize(frame,  (width, height))
#            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            stop = round(time.time() * 1000)
            
            print("-------TIMING--------")
            print("resize image: {} ms".format(stop - start))

            start = round(time.time() * 1000)
            top_result = process_image(interpreter, image, input_index)
            stop = round(time.time() * 1000)

            print("process image: {} ms".format(stop - start))
            
            print("-------RESULTS--------")
            for obj in top_result:
                pos = obj["pos"]
                _id = obj["_id"]
                score = obj["score"] * 100
                x1 = int(pos[1] * CAMERA_WIDTH)
                x2 = int(pos[3] * CAMERA_WIDTH)
                y1 = int(pos[0] * CAMERA_HEIGHT)
                y2 = int(pos[2] * CAMERA_HEIGHT)
                print("class: {}, x1: {}, y1: {}, x2: {}, y2: {}, score: {}".format(labels[_id], x1, y1, x2, y2, score))

            if SHOW_IMAGE:
                display_result(top_result, frame, labels)

                c = cv2.waitKey(1)
                if c == 27:
                    break


if __name__ == "__main__":
   main(sys.argv[1:])
