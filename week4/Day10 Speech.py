import speech_recognition as sr

a = sr.AudioFile("C:/Users/11453/PycharmProjects/riskassessment/data/OSR_us_000_0012_8k.wav")
with a as source:
    a = sr.Recognizer().record(source)
print(sr.Recognizer().recognize_google(a))

r = sr.Recognizer()

print("please say something in 4 seconds... and wait for 4 seconds for the answer.....")
print("Accessing Microphone..")

with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source, duration=2)
    print("Waiting for you to speak...")
    audio = r.listen(source)
try:
    print("You said " + r.recognize_google(audio))
except:
    print("Please retry...")