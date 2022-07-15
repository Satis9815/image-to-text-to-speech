
from flask import Flask,render_template,Response,request,redirect
import cv2
import pytesseract
import googletrans
import numpy as np
import secrets
import gtts
from googletrans import Translator
import pyttsx3

# from requests import request

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            img=cv2.imwrite("image.jpg",frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


LANGUAGES = {
    'en': 'english',
    'hi': 'hindi',
    'kn': 'kannada',
    'ne': 'nepali'
}

@app.route('/camera')
def index():
    return render_template('index.html')

@app.route("/getCapturedImage",methods=["GET","POST"])
def getCapturedImage():
    manage=False
    if request.method=="POST":
        manage=True
        select=request.form.get('lang_list')
        print(select)
        imgFile=cv2.imread("image.jpg")
        def noise_removal(image):
            import numpy as np
            kernel = np.ones((2, 2), np.uint8)
            image = cv2.dilate(image, kernel, iterations=1)
            kernel = np.ones((2, 2), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE,kernel)
            image = cv2.medianBlur(image,3)
            return image
        def normalization(img):
            resultimage = np.zeros((800, 800))
            normalizedimage = cv2.normalize(no_noise,resultimage, 0, 100, cv2.NORM_MINMAX)
            return normalizedimage

        def grayscale(image):
            gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            return gray_img
        def thresholding(gray_image):#thresholding/binarizalation
            thresh,imbw =cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return imbw
        # preprocessing the images 
        no_noise = noise_removal(imgFile)
        normal_img = normalization(no_noise)
        gray_image = grayscale(normal_img)
        thre_img=thresholding(gray_image)
        text = pytesseract.image_to_string(thre_img,lang="blur")
        print(text)
        
        translater = googletrans.Translator()
        text_to_translate = text
        print(len(text_to_translate))
        if len(text_to_translate) !=0:
            out = translater.translate(text_to_translate, dest=select)
            translated_text = out.text
            translated = translated_text

            #Savein  mp3 format file of a translated text
            obj = gtts.gTTS(text = translated, slow=False, lang = select)
            adfiile=secrets.token_hex(5)
            obj.save("./static/"+adfiile)       
            return render_template("data.html",data=LANGUAGES,selected_lang=select,
        extracted_text=text,translated_texts=translated_text,datamanage=manage,camaudiofile=adfiile)
        else:
            engine = pyttsx3.init()
            engine.say('Please choose a image again containing text.')
            engine.runAndWait()
            return redirect('camera')
    return render_template("data.html",data=LANGUAGES)
@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
