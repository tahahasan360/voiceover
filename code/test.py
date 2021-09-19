# Import the required module for text 
# to speech conversion
from gtts import gTTS
from playsound import playsound
  
# This module is imported so that we can 
# play the converted audio
import os
  
# The text that you want to convert to audio
mytext = 'In'
  
# Language in which you want to convert
language = 'en'
  
playsound("cached_tts/hello.mp3")
