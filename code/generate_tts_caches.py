from gtts import gTTS
from playsound import playsound
from constants import word_dict

language = 'en'

for i in word_dict:
    text = word_dict[i].lower()
    tts = gTTS(text=text, lang=language, slow=True)
    tts.save("cached_tts/" + text + ".mp3")

tts = gTTS(text="2021", lang=language, slow=True)
tts.save("cached_tts/2021.mp3")



