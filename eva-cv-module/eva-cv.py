# Software developed by Marcelo Marques da Rocha
# MidiaCom Laboratory - Universidade Federal Fluminense
# This work was funded by CAPES and Google Research

import argparse
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager

sys.path.append("/home/pi/EVA_ROBOT")

import config
import cv2
import face_recognition as fr
import numpy as np
from paho.mqtt import client as mqtt_client
from picamera import PiCamera
from picamera.array import PiRGBArray
from pyzbar import pyzbar
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

broker = config.MQTT_BROKER_ADRESS
port = config.MQTT_PORT
topic_base = config.EVA_TOPIC_BASE

TOPIC_EMOTION = topic_base + "/USEREMOTION"
TOPIC_USER_ID = topic_base + "/USERID"
TOPIC_QR_READ = topic_base + "/QRREAD"

TOPIC_EMOTION_RESPONSE = topic_base + "/USEREMOTION_RESPONSE"
TOPIC_USER_ID_RESPONSE = topic_base + "/USERID_RESPONSE"
TOPIC_QR_RESPONSE = topic_base + "/QRREAD_RESPONSE"
TOPIC_SYSLOG = topic_base + "/syslog"

CASCADE_PATH = "eva-cv-module/haarcascade_frontalface_default.xml"
MODEL_PATH = "eva-cv-module/model.h5"
USERS_PATH = "eva-cv-module/users"

CAMERA_SETTLE_TIME = 0.3


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "-f",
    "--fullscreen",
    help="Display window in full screen",
    action="store_true",
)

parser.add_argument(
    "-v",
    "--video",
    help="Display video capture window",
    action="store_true",
)

parser.add_argument(
    "-fl",
    "--flip",
    help="Flip incoming video signal",
    action="store_true",
)

args = parser.parse_args()


# ----------------------------------------------------------------------
# Expression recognition model
# ----------------------------------------------------------------------

model = Sequential()

model.add(
    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(48, 48, 1),
    )
)

model.add(
    Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(
    Conv2D(
        128,
        kernel_size=(3, 3),
        activation="relu",
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(
        128,
        kernel_size=(3, 3),
        activation="relu",
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax"))

model.load_weights(MODEL_PATH)

emotion_dict = {
    0: "ANGRY",
    1: "DISGUST",
    2: "FEAR",
    3: "HAPPY",
    4: "NEUTRAL",
    5: "SAD",
    6: "SURPRISE",
}


# ----------------------------------------------------------------------
# Face detector
# ----------------------------------------------------------------------

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError(
        "Unable to load facial cascade: " + CASCADE_PATH
    )


# ----------------------------------------------------------------------
# MQTT client
# ----------------------------------------------------------------------

client = mqtt_client.Client()


# ----------------------------------------------------------------------
# Capture control
# ----------------------------------------------------------------------

# Sempre contém apenas a solicitação mais recente.
latest_request = None

# É incrementado em toda nova solicitação.
# Uma captura antiga detecta que sua geração não é mais válida e encerra.
request_generation = 0

request_condition = threading.Condition()
stop_event = threading.Event()


def capture_was_cancelled(generation):
    with request_condition:
        return (
            stop_event.is_set()
            or generation != request_generation
        )


def publish_if_current(generation, topic, message):
    """
    Publica somente se a captura ainda for a solicitação atual.

    Isso impede que uma captura cancelada publique uma resposta atrasada.
    """

    with request_condition:
        if (
            stop_event.is_set()
            or generation != request_generation
        ):
            return False

        client.publish(topic, message)
        return True


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def publish_log(message):
    print(message)
    client.publish(
        TOPIC_SYSLOG,
        "Computer Vision: " + message,
    )


def prepare_image(image):
    if image is None or image.size == 0:
        raise RuntimeError("Camera returned an empty image.")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise RuntimeError(
            "Invalid camera image shape: " + str(image.shape)
        )

    if args.flip:
        image = cv2.flip(image, -1)

    return np.ascontiguousarray(image)


def reset_buffer(raw_capture):
    raw_capture.seek(0)
    raw_capture.truncate(0)


def show_video(window_name, image):
    if not args.video:
        return True

    cv2.imshow(window_name, image)

    if args.fullscreen:
        cv2.setWindowProperty(
            window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )

    key = cv2.waitKey(1) & 0xFF
    return key != ord("q")


@contextmanager
def open_camera_stream(camera, resolution, framerate):
    """
    Abre um stream e garante que ele seja fechado ao terminar,
    inclusive quando uma captura é cancelada ou gera uma exceção.
    """

    raw_capture = None
    stream = None

    try:
        camera.resolution = resolution
        camera.framerate = framerate

        time.sleep(CAMERA_SETTLE_TIME)

        raw_capture = PiRGBArray(
            camera,
            size=resolution,
        )

        stream = camera.capture_continuous(
            raw_capture,
            format="bgr",
            use_video_port=True,
        )

        yield stream, raw_capture

    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception:
                traceback.print_exc()

        if raw_capture is not None:
            try:
                raw_capture.close()
            except Exception:
                traceback.print_exc()

        if args.video:
            cv2.destroyAllWindows()
            cv2.waitKey(1)


# ----------------------------------------------------------------------
# Expression recognition
# ----------------------------------------------------------------------

def capture_emotion(camera, generation):
    if capture_was_cancelled(generation):
        return

    publish_log("Eva will try to identify the user emotion.")

    with open_camera_stream(
        camera,
        resolution=(640, 480),
        framerate=10,
    ) as (stream, raw_capture):

        for frame in stream:
            try:
                # Encerra sem responder caso outra solicitação tenha chegado.
                if capture_was_cancelled(generation):
                    print("Emotion capture cancelled.")
                    return

                image = prepare_image(frame.array)
                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY,
                )

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(30, 30),
                )

                for x, y, width, height in faces:
                    if capture_was_cancelled(generation):
                        return

                    x = max(0, x)
                    y = max(0, y)

                    x2 = min(
                        gray.shape[1],
                        x + width,
                    )

                    y2 = min(
                        gray.shape[0],
                        y + height,
                    )

                    roi_gray = gray[y:y2, x:x2]

                    if roi_gray.size == 0:
                        continue

                    resized_face = cv2.resize(
                        roi_gray,
                        (48, 48),
                        interpolation=cv2.INTER_AREA,
                    )

                    model_input = np.expand_dims(
                        np.expand_dims(
                            resized_face,
                            axis=-1,
                        ),
                        axis=0,
                    )

                    prediction = model.predict(
                        model_input,
                        verbose=0,
                    )

                    if capture_was_cancelled(generation):
                        return

                    emotion_index = int(
                        np.argmax(prediction)
                    )

                    emotion_label = emotion_dict[
                        emotion_index
                    ]

                    if not publish_if_current(
                        generation,
                        TOPIC_EMOTION_RESPONSE,
                        emotion_label,
                    ):
                        return

                    print(
                        "Inferred user emotion:",
                        emotion_label,
                    )

                    client.publish(
                        TOPIC_SYSLOG,
                        "Inferred User Emotion: "
                        + emotion_label,
                    )

                    if args.video:
                        cv2.rectangle(
                            image,
                            (x, y),
                            (x2, y2),
                            (255, 0, 0),
                            2,
                        )

                        cv2.putText(
                            image,
                            emotion_label,
                            (x, max(30, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        show_video(
                            "Emotion recognition",
                            image,
                        )

                    return

                if not show_video(
                    "Emotion recognition",
                    image,
                ):
                    print("Emotion capture cancelled by user.")
                    return

            finally:
                reset_buffer(raw_capture)


# ----------------------------------------------------------------------
# Known users
# ----------------------------------------------------------------------

def load_known_users():
    known_users = []

    if not os.path.isdir(USERS_PATH):
        raise RuntimeError(
            "Users directory does not exist: " + USERS_PATH
        )

    for file_name in os.listdir(USERS_PATH):
        if not file_name.lower().endswith(".npy"):
            continue

        file_path = os.path.join(
            USERS_PATH,
            file_name,
        )

        try:
            encoding = np.asarray(
                np.load(file_path)
            ).reshape(-1)

            if encoding.size != 128:
                print(
                    "Ignoring invalid encoding:",
                    file_name,
                )
                continue

            user_id = file_name.split("_")[0]

            known_users.append(
                (
                    user_id,
                    encoding,
                    file_name,
                )
            )

        except Exception:
            print(
                "Unable to load user encoding:",
                file_name,
            )
            traceback.print_exc()

    return known_users


# ----------------------------------------------------------------------
# Face recognition
# ----------------------------------------------------------------------

def capture_user_id(camera, generation):
    if capture_was_cancelled(generation):
        return

    publish_log("Eva will try to identify the user.")

    known_users = load_known_users()

    with open_camera_stream(
        camera,
        resolution=(400, 400),
        framerate=10,
    ) as (stream, raw_capture):

        for frame in stream:
            try:
                if capture_was_cancelled(generation):
                    print("Face recognition cancelled.")
                    return

                image = prepare_image(frame.array)

                # face_recognition utiliza RGB.
                rgb_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB,
                )

                face_locations = fr.face_locations(
                    rgb_image
                )

                if face_locations:
                    # Seleciona o maior rosto da imagem.
                    face_locations.sort(
                        key=lambda location: (
                            location[2] - location[0]
                        ) * (
                            location[1] - location[3]
                        ),
                        reverse=True,
                    )

                    selected_location = face_locations[0]

                    encodings = fr.face_encodings(
                        rgb_image,
                        [selected_location],
                    )

                    if encodings:
                        current_encoding = encodings[0]

                        user_id = "unknown"
                        minimum_distance = float("inf")

                        for (
                            known_user_id,
                            known_encoding,
                            file_name,
                        ) in known_users:

                            if capture_was_cancelled(
                                generation
                            ):
                                return

                            distance = float(
                                fr.face_distance(
                                    [known_encoding],
                                    current_encoding,
                                )[0]
                            )

                            print(
                                "Comparing current user with:",
                                file_name,
                                "Distance:",
                                distance,
                            )

                            if distance < minimum_distance:
                                minimum_distance = distance
                                user_id = known_user_id

                        if minimum_distance > 0.42:
                            user_id = "unknown"

                        if not publish_if_current(
                            generation,
                            TOPIC_USER_ID_RESPONSE,
                            user_id,
                        ):
                            return

                        client.publish(
                            TOPIC_SYSLOG,
                            "User ID: " + user_id,
                        )

                        print("User ID:", user_id)
                        return

                if not show_video(
                    "Face recognition",
                    image,
                ):
                    print(
                        "Face recognition cancelled by user."
                    )
                    return

            finally:
                reset_buffer(raw_capture)


# ----------------------------------------------------------------------
# QR Code reader
# ----------------------------------------------------------------------

def capture_qr_code(camera, generation):
    if capture_was_cancelled(generation):
        return

    publish_log("Eva will try to read a QR Code.")

    with open_camera_stream(
        camera,
        resolution=(640, 480),
        framerate=30,
    ) as (stream, raw_capture):

        for frame in stream:
            try:
                if capture_was_cancelled(generation):
                    print("QR Code capture cancelled.")
                    return

                image = prepare_image(frame.array)

                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY,
                )

                # Evita problemas com memória não contínua no pyzbar.
                gray = np.ascontiguousarray(gray)

                barcodes = pyzbar.decode(gray)

                for barcode in barcodes:
                    if capture_was_cancelled(generation):
                        return

                    try:
                        decoded_text = barcode.data.decode(
                            "utf-8",
                            errors="strict",
                        )
                    except UnicodeDecodeError:
                        decoded_text = barcode.data.decode(
                            "utf-8",
                            errors="replace",
                        )

                    if not publish_if_current(
                        generation,
                        TOPIC_QR_RESPONSE,
                        decoded_text,
                    ):
                        return

                    print(
                        "QR Code content:",
                        decoded_text,
                    )

                    client.publish(
                        TOPIC_SYSLOG,
                        "QR Code content: " + decoded_text,
                    )

                    return

                if not show_video(
                    "QR Code reader",
                    image,
                ):
                    print(
                        "QR Code reading cancelled by user."
                    )
                    return

            finally:
                reset_buffer(raw_capture)


# ----------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------

def publish_error_if_current(generation, topic):
    if topic == TOPIC_EMOTION:
        response_topic = TOPIC_EMOTION_RESPONSE
    elif topic == TOPIC_USER_ID:
        response_topic = TOPIC_USER_ID_RESPONSE
    elif topic == TOPIC_QR_READ:
        response_topic = TOPIC_QR_RESPONSE
    else:
        return

    publish_if_current(
        generation,
        response_topic,
        "ERROR",
    )


# ----------------------------------------------------------------------
# Camera worker
# ----------------------------------------------------------------------

def vision_worker():
    global latest_request

    camera = None

    try:
        # A câmera é criada e utilizada sempre pela mesma thread.
        camera = PiCamera()

        time.sleep(CAMERA_SETTLE_TIME)

        publish_log("The camera was initialized.")

        while not stop_event.is_set():
            with request_condition:
                while (
                    latest_request is None
                    and not stop_event.is_set()
                ):
                    request_condition.wait()

                if stop_event.is_set():
                    break

                # Obtém somente a solicitação mais recente.
                topic, generation = latest_request
                latest_request = None

            try:
                if topic == TOPIC_EMOTION:
                    capture_emotion(
                        camera,
                        generation,
                    )

                elif topic == TOPIC_USER_ID:
                    capture_user_id(
                        camera,
                        generation,
                    )

                elif topic == TOPIC_QR_READ:
                    capture_qr_code(
                        camera,
                        generation,
                    )

            except Exception as error:
                traceback.print_exc()

                # Não publica erro para uma captura que foi substituída.
                if not capture_was_cancelled(generation):
                    publish_log(
                        "Capture operation failed: "
                        + str(error)
                    )

                    publish_error_if_current(
                        generation,
                        topic,
                    )

    except Exception as error:
        traceback.print_exc()

        publish_log(
            "Camera initialization failed: "
            + str(error)
        )

    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:
                traceback.print_exc()

        cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# MQTT callbacks
# ----------------------------------------------------------------------

def on_connect(client_instance, userdata, flags, rc):
    if rc != 0:
        print(
            "MQTT connection failed. Code:",
            rc,
        )
        return

    client_instance.subscribe(
        [
            (TOPIC_EMOTION, 1),
            (TOPIC_USER_ID, 1),
            (TOPIC_QR_READ, 1),
        ]
    )

    print("Computer Vision Module - Connected.")


def on_message(client_instance, userdata, msg):
    """
    Substitui imediatamente a operação anterior pela nova solicitação.

    Não adiciona mensagens em fila e não acessa diretamente a câmera.
    """

    global latest_request
    global request_generation

    if msg.topic not in (
        TOPIC_EMOTION,
        TOPIC_USER_ID,
        TOPIC_QR_READ,
    ):
        return

    with request_condition:
        # Cancela logicamente qualquer captura em andamento.
        request_generation += 1

        # Sobrescreve qualquer solicitação pendente.
        latest_request = (
            msg.topic,
            request_generation,
        )

        request_condition.notify()

    print(
        "New camera request:",
        msg.topic,
    )


# ----------------------------------------------------------------------
# Application initialization
# ----------------------------------------------------------------------

client.on_connect = on_connect
client.on_message = on_message

camera_thread = threading.Thread(
    target=vision_worker,
    name="vision-worker",
    daemon=True,
)

camera_thread.start()

try:
    client.connect(
        broker,
        port,
    )

    client.loop_forever()

except KeyboardInterrupt:
    print("Stopping Computer Vision Module.")

except Exception:
    traceback.print_exc()
    print("Unable to connect to Broker.")

finally:
    stop_event.set()

    with request_condition:
        request_condition.notify_all()

    client.disconnect()
    cv2.destroyAllWindows()