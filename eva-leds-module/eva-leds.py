# Software developed by Marcelo Marques da Rocha
# MidiaCom Laboratory - Universidade Federal Fluminense
# This work was funded by CAPES and Google Research

from paho.mqtt import client as mqtt_client

import os
import signal
import subprocess
import sys

sys.path.append('/home/pi/EVA_ROBOT')
import config

broker = config.MQTT_BROKER_ADRESS
port = config.MQTT_PORT
topic_base = config.EVA_TOPIC_BASE

animation_dir = "eva-leds-module/leds-animation/"
animations = {
    "ANGRY": "angry",
    "ANGRY2": "angry2",
    "INLOVE": "angry",
    "HAPPY": "happy",
    "LISTEN": "listen",
    "PROCESS": "process",
    "RAINBOW": "rainbow",
    "SAD": "sad",
    "FEAR": "sad",
    "SAD2": "sad2",
    "SURPRISE": "surprise",
    "DISGUST": "surprise",
    "SPEAK": "speak",
    "WHITE": "white",
}

p = None


def stop_animation(turn_off=False):
    global p

    if p and p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    p = None

    if turn_off:
        os.system(animation_dir + "stop")


def shutdown(signum, frame):
    sys.exit(0)


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(topic=[(topic_base + '/LEDS', 1)])


def on_message(client, userdata, msg):
    global p

    command = msg.payload.decode()

    if command == "STOP":
        stop_animation(turn_off=True)
        return

    if command not in animations:
        return

    client.publish(topic_base + '/syslog', "Leds Animation: " + command)
    stop_animation()
    p = subprocess.Popen(
        animation_dir + animations[command],
        stdout=subprocess.PIPE,
        shell=True,
        preexec_fn=os.setsid
    )


signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

client = mqtt_client.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(broker, port)
except:
    print("Unable to connect to Broker.")
    sys.exit(1)

try:
    client.loop_forever()
finally:
    stop_animation(turn_off=True)