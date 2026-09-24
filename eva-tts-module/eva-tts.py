# Software developed by Marcelo Marques da Rocha
# MidiaCom Laboratory - Universidade Federal Fluminense
# This work was funded by CAPES and Google Research

import hashlib
import os
from paho.mqtt import client as mqtt_client

import time

from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import ApiException


import sys
sys.path.append('/home/pi/EVA_ROBOT')
import config # Module with network device configurations.

broker = config.MQTT_BROKER_ADRESS # Broker address.
port = config.MQTT_PORT # Broker Port.
topic_base = config.EVA_TOPIC_BASE

voice_tone = config.VOICE_TONE


# Watson API configuration key.
with open("eva-tts-module/ibm_cred.txt", "r") as ibm_cred: 
    ibm_config = ibm_cred.read().splitlines()
apikey = ibm_config[0]
url = ibm_config[1]

# Setup Watson service.
authenticator = IAMAuthenticator(apikey)

# TTS service.
tts = TextToSpeechV1(authenticator = authenticator)
tts.set_service_url(url)

auth_start_time = time.time() # Time of authentication.
first_requisition = True; # Indicates that this is the first request for the Watson service.

# MQTT
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    # Subscribing in on_connect() means that if we lose the connection and
    # Reconnect then subscriptions will be renewed.
    client.subscribe(topic=[(topic_base + '/TALK', 1), ])
    print("Text-To-Speech Module - Connected.")
    
# The callback for when a PUBLISH message is received from the server.
# Any failure (Watson quota exhausted, no internet, invalid credentials...) must not kill the module:
# it is logged and the TALK_RESPONSE is sent anyway, so the simulator does not wait forever.
def on_message(client, userdata, msg):
    try:
        talk(client, msg)
    except Exception as ex:
        if isinstance(ex, ApiException):
            error = "Watson error code " + str(ex.code) + ": " + str(ex.message)
        else:
            error = type(ex).__name__ + ": " + str(ex)
        print("The text could not be spoken. " + error)
        client.publish(topic_base + "/syslog", "EVA could not speak the text (" + error + ")")
        client.publish(topic_base + "/TALK_RESPONSE") # Releases the robot even without speech.


def talk(client, msg):
    global voice_tone, auth_start_time, apikey, url, authenticator, tts, first_requisition
    if msg.topic == topic_base + '/TALK':
        print("Using IBM Watson to convert text to audio...")
        # Assumes the default UTF-8 (Generates the hashing of the audio file).
        # Additionally, use the voice timbre attribute in the file hash.
        if len(msg.payload.decode().split("|")) == 2:
            voice_tone = msg.payload.decode().split("|")[0]
            msg.payload = (msg.payload.decode()).split("|")[1]
            msg.payload = msg.payload.encode()

        print("Voice:", voice_tone, "Message:", msg.payload.decode())
        hash_object = hashlib.md5(msg.payload)
        file_name = "_audio_"  + voice_tone + hash_object.hexdigest()
        
        audio_file_is_ok = False
        while(not audio_file_is_ok):
            # Checks if the speech audio already exists in the cache folder.
            if not (os.path.isfile("eva-tts-module/tts_cache_files/" + file_name + config.WATSON_AUDIO_EXTENSION)): # If it doesn't exist, call Watson.
                print("The file is not cached... Let's try to generate it!")

                # Tests with the first request after a module reset (There is no problem in this way).
                # ================================================== ======================================
                # Watson appears to impose a limited connection time from the first request.
                # 1min of inactivity -> OK
                # 2min of inactivity -> OK
                # 8min of inactivity -> OK

                # Tests with one request from another, without resetting the module.
                # ================================================== ======================================
                # 3min of inactivity from the first request -> OK
                # 3.30min of inactivity from the first request -> OK
                # 4min of inactivity from a first req. -> Crashed! (Request rejected)
                # 5min of inactivity from the first request.-> Crashed! (Request rejected)

                # Checks if the module has been inactive for more than 3min (180s) since the first request.
                print(first_requisition, (time.time() - auth_start_time))
                if (not(first_requisition) and (time.time() - auth_start_time >= 180)):
                    print("The module has been inactive for more than 3 minutes (since the first request) and a new authentication will be performed.")
                    # Wtson API key (config.)
                    with open("eva-tts-module/ibm_cred.txt", "r") as ibm_cred: 
                        ibm_config = ibm_cred.read().splitlines()
                    apikey = ibm_config[0]
                    url = ibm_config[1]
                    # Setup Watson service
                    authenticator = IAMAuthenticator(apikey)
                    # TTS service
                    tts = TextToSpeechV1(authenticator = authenticator)
                    tts.set_service_url(url)
                    first_requisition = False
                    auth_start_time = time.time() # Momento da autenticação.

                # Start the TTS process
                tts_start = time.time() # Variable used to mark the processing time of the TTS service.
                # Calls Watson BEFORE creating the file, so a failure does not leave an empty (0 bytes) file in the cache.
                # A failure raises an exception that is handled in on_message().
                res = tts.synthesize(msg.payload.decode(), accept = config.ACCEPT_AUDIO_EXTENSION, voice = voice_tone).get_result()
                if not res.content: # Corrupted audio!
                    raise RuntimeError("Watson returned an empty audio.")
                print("Writing content to disk...")
                with open("eva-tts-module/tts_cache_files/" + file_name + config.WATSON_AUDIO_EXTENSION, 'wb') as audio_file:
                    audio_file.write(res.content)
                tts_ending = time.time()
                client.publish(topic_base + "/syslog", "The audio was generated correctly in (s): %.2f" % (tts_ending - tts_start))
                print("The file will be played!")
                client.publish(topic_base + "/syslog", "EVA is busy trying to speak the text: " + msg.payload.decode())
                client.publish(topic_base + "/speech", file_name)
                audio_file_is_ok = True
                first_requisition = False
            else:
                print("The file is cached!")
                if (os.path.getsize("eva-tts-module/tts_cache_files/" + file_name + config.WATSON_AUDIO_EXTENSION)) == 0: # Corrupted file
                    print("The generated audio file is 0 bytes, corrupt and will be removed!")
                    os.remove("eva-tts-module/tts_cache_files/" + file_name + config.WATSON_AUDIO_EXTENSION)
                else:
                    print("The file is more than 0 bytes and will be played now!")
                    client.publish(topic_base + "/syslog", "The audio was found in the cache.")
                    client.publish(topic_base + "/syslog", "EVA is busy trying to speak the text: " + msg.payload.decode())
                    client.publish(topic_base + "/speech", file_name)
                    audio_file_is_ok = True  
        
        
        
# Run the MQTT client thread.
client = mqtt_client.Client()
client.on_connect = on_connect
client.on_message = on_message
try:
    client.connect(broker, port)
except:
    print ("Unable to connect to Broker.")
    exit(1)


client.loop_forever()

