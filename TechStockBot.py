#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:16:49 2020

@author: RishiSingh
"""
##Libraries used
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
import webbrowser
import re
import json
import csv
import pyttsx3
from io import StringIO
from bs4 import BeautifulSoup
import requests
import random
import os
import csv
import nltk
import aiml
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image,ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
#######################################################
#  Initialise NLTK Inference
#######################################################
from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring


#######################################################
#  Azure
#######################################################

import os
from azure.cognitiveservices.language.textanalytics import TextAnalyticsClient
from msrest.authentication import CognitiveServicesCredentials
import os
import IPython
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import os




#######################################################
#  Initialise Knowledgebase. 
#######################################################
import pandas
kb=[]
data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
# >>> ADD SOME CODES here for checking KB integrity (no contradiction), 
# otherwise show an error message and terminate

##GUI root
root = Tk()

#######################################################
#  Initialise AIML agent
#######################################################

# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the fgiles are loaded.
# The optional brainFile argument specifies a brain file to load.
kern.bootstrap(learnFiles="coursework.xml")
#######################################################


#######################################################
# Keys for Azure
cog_key = 'c3a090e46a954bf7ba1a52ec4af527c8'
cog_endpoint = 'https://n0834113ait4.cognitiveservices.azure.com/'
cog_region = 'uksouth'
#######################################################


#######################################################
# Main loop
#######################################################

url_profile = "https://finance.yahoo.com/quote/{}/profile?p={}" ##URL for web scrapping
url_price = 'https://uk.finance.yahoo.com/quote/{}?p={}'

# engine = pyttsx3.init()  
# voices = engine.getProperty('voices') 
# engine.setProperty('voice', voices[0].id)
#Text to speach does not work fully when using GUI but it does when using console.
file = open('TechQA.csv','r',errors = 'ignore') # Data from CSV file.
brandPath = " "

corpus = file.read()
corpus = corpus.lower()
nltk.download('punkt') 
nltk.download('wordnet')
##Preprocessing data
sent_tokens = nltk.sent_tokenize(corpus)# converts to list of sentences 
word_tokens = nltk.word_tokenize(corpus)# converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()
#class names for CNN
class_names = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari', 'Ford', 'Google', 'HP', 'Heineken', 'Intel', 'McDonalds', 'Mini', 'Multiple', 'Nbc', 'Nike', 'None', 'Pepsi', 'Porsche', 'Puma', 'Redbull', 'Sprite', 'Starbucks', 'Texaco', 'Unicef', 'Vodafone', 'Yahoo']
def LemTokens(tokens): 
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def gen_response(user_response):  ## Code adapted from https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e
    response_bot = " "
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') #Vectorising
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf) #cosine similarity 
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        return response_bot
    else:
        response_bot = response_bot+sent_tokens[idx]
        return response_bot
    
def clear():  #Clears GUI text
    txt.delete(1.0,END)
    
def openImage(imageName): #Opens image
    new_window = Toplevel(root)
    # get image
    image = ImageTk.PhotoImage(Image.open("/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/AI_N0834113(2)/IMAGES/"+imageName+".png")) #directory must be changed get correct picture
    # load image
    panel = Label(new_window, image=image)
    panel.image = image
    panel.pack()
    

def browsefile():#Select file function
     root.filename = filedialog.askopenfilename(initialdir="/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/AI_N0834113(2)/Logos/Predictions", title="Select A File", filetypes=(("jpg files", "*.jpg"),("all files","*.*")))
     return root.filename#return file name
    
def quit(): #To quit GUI
    root.destroy()
    
def cnnModel(filename1, code): #CNN adapted from tensorflow
    model = load_model((os.path.join(os.getcwd(),"/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/AI_N0834113(2)/BrandsLogoCNNFinal.h5")))
    model.summary() #model loaded and summary generated 
    image_path = filename1 
    img = keras.preprocessing.image.load_img( image_path, target_size=(180, 180)) #image preprocessing 
    img_array = keras.preprocessing.image.img_to_array(img) 
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])#compile the model
    predictions = model.predict(img_array) # predictions made
    score = tf.nn.softmax(predictions[0])
    answer = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    answer = translate_text(cog_region, cog_key, answer, to_lang=code, from_lang="en")
    txt.insert(END,"\n" + "Bot: " + answer)
    class_name1 = "{}".format(class_names[np.argmax(score)]) # class found of image
    im = Image.open(filename1)
    im.show() #open image
    return class_name1 # return image name

def cnnModelChosen(imagefileName, code): #cnn with image file 
    model = load_model((os.path.join(os.getcwd(),"/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/AI_N0834113(2)/BrandsLogoCNNFinal.h5")))
    model.summary()
    image_path = "/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/flickr_logos_27_dataset 2/Logos/Predictions/" + imagefileName + ".jpg"
    img = keras.preprocessing.image.load_img( image_path, target_size=(180, 180)) 
    img_array = keras.preprocessing.image.img_to_array(img) 
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    predictions = model.predict(img_array) 
    score = tf.nn.softmax(predictions[0])
    answer = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    answer = translate_text(cog_region, cog_key, answer, to_lang=code, from_lang="en")
    print(answer)
    txt.insert(END,"\n" + "Bot: " + answer)
    im = Image.open(image_path)
    im.show()
    
def vgg16Model(filename): #pretrained VGG16 model
    from keras.preprocessing.image import ImageDataGenerator 
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) 
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory('/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/AI_N0834113(2)/Logos/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical') 
    test_set = test_datagen.flow_from_directory('/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/flickr_logos_27_dataset 2/Logos/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
    
    model = load_model((os.path.join(os.getcwd(),"/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/AI_N0834113(2)/BrandsVGG16.h5")))
    model.summary()
    image_path = filename
    img = keras.preprocessing.image.load_img(
    image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    predicted_class_indices=np.argmax(predictions,axis=1)
    labels = (training_set.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions1 = [labels[k] for k in predicted_class_indices]
    im = Image.open(filename)
    im.show() #open image
    txt.insert(END,"\n" + "Bot: The VGG16 model predicts this model is " + predictions1[0])

def ResNet50Model(filename): #pretrained Resnet model
    from keras.preprocessing.image import ImageDataGenerator 
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) 
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory('/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/flickr_logos_27_dataset 2/Logos/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical') 
    test_set = test_datagen.flow_from_directory('/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/AI_N0834113(2)/Logos/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
    
    model = load_model((os.path.join(os.getcwd(),"/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/AI_N0834113(2)/BrandsVGG16.h5")))
    model.summary()
    image_path = filename
    img = keras.preprocessing.image.load_img(
    image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    predicted_class_indices=np.argmax(predictions,axis=1)
    labels = (training_set.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions1 = [labels[k] for k in predicted_class_indices]
    im = Image.open(filename)
    im.show() #open image
    txt.insert(END,"\n" + "Bot: The ResNet50 model predicts this model is " + predictions1[0])
    
def MobileNet(filename): #pretrained Inception model
    from keras.preprocessing.image import ImageDataGenerator 
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) 
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory('/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/flickr_logos_27_dataset 2/Logos/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical') 
    test_set = test_datagen.flow_from_directory('/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/AI_N0834113(2)/Logos/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
    
    model = load_model((os.path.join(os.getcwd(),"/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/AI_N0834113(2)/BrandsMobile.h5")))
    model.summary()
    image_path = filename
    img = keras.preprocessing.image.load_img(
    image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    predicted_class_indices=np.argmax(predictions,axis=1)
    labels = (training_set.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions1 = [labels[k] for k in predicted_class_indices]
    im = Image.open(filename)
    im.show() #open image
    txt.insert(END,"\n" + "Bot: The MobileNet model predicts this model is " + predictions1[0])
    
    
def brandNews(classNames,code): # News api which gets news from the class chosen by cnn
    urlNews = "https://newsapi.org/v2/everything?q=" + classNames + "&from=2021-03-26&to=2021-03-26&sortBy=popularity&apiKey=6c2ceae5f916435183851b58debd63de"
    open_page = requests.get(urlNews).json() #open page of news api
    new_answer = "Let me give you some extra information"
    new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
    txt.insert(END,"\n" + "Bot: " + new_answer)           
    article = open_page["articles"]  
    resultsURL = [] 
    resultsTitle = []
    from random import randint #random number generator 
    for _ in range(1):
        value = randint(0, 10)
      
    for ar in article: 
        resultsTitle.append(ar["title"]) # get title and URL for news
        resultsURL.append(ar["url"]) 
          
    for i in range(len(resultsURL)):  #generate a random news at a certain index
        if i == value:
            new_answer = "Here is the top headline about "+ classNames + ": " + resultsTitle[value]
            new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
            txt.insert(END,"\n" + new_answer)
            new_answer = "Here is the top headline about "+ classNames + ": " + resultsTitle[value]
            new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
            txt.insert(END,"\n" + new_answer) 
            webbrowser.open(resultsURL[value]) #web page is opened                                 
            break
        
def detect_lang(userInput):
    # Get a client for your text analytics cognitive service resource
    
    texts = []
    user = "user"
    userText = userInput
    text = {"id":user,"text":userText}
    texts.append(text)
    for text_num in range(len(texts)):
        print('{}\n{}\n'.format(texts[text_num]['id'], texts[text_num]['text']))

    # Get a client for your text analytics cognitive service resource
    text_analytics_client = TextAnalyticsClient(endpoint=cog_endpoint,credentials=CognitiveServicesCredentials(cog_key))
    
    # Analyze the reviews you read from the /data/reviews folder earlier
    language_analysis = text_analytics_client.detect_language(documents=texts)
    
    # print detected language details for each review
    for text_num in range(len(texts)):
    # print the review id
        print(texts[text_num]['id'])

        # Get the language details for this review
        lang = language_analysis.documents[text_num].detected_languages[0]
        print(' - Language: {}\n - Code: {}\n - Score: {}\n'.format(lang.name, lang.iso6391_name, lang.score))

        return lang.iso6391_name

def translate_text(cog_region, cog_key, text, to_lang='en', from_lang='en'):
    import requests, uuid, json

    # Create the URL for the Text Translator service REST request
    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    params = '&from={}&to={}'.format(from_lang, to_lang)
    constructed_url = path + params

    # Prepare the request headers with Cognitive Services resource key and region
    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region':cog_region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # Add the text to be translated to the body
    body = [{
        'text': text
    }]

    # Get the translation
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    print(response[0]["translations"][0]["text"])
    return response[0]["translations"][0]["text"]

def article_review(fileName):
    # Read the reviews in the /data/reviews folder
    reviews_folder = os.path.join('data', 'text', 'reviews')
    reviews = []
    review_text = open(os.path.join(reviews_folder, fileName + ".txt")).read()
    review = {"id": fileName, "text": review_text}
    reviews.append(review)

    # Get a client for your text analytics cognitive service resource
    text_analytics_client = TextAnalyticsClient(endpoint=cog_endpoint,
                                            credentials=CognitiveServicesCredentials(cog_key))
        
        # # Use the client and reviews you created in the previous code cell to get key phrases
    key_phrase_analysis = text_analytics_client.key_phrases(documents=reviews)

    # print key phrases for each review
    for review_num in range(len(reviews)):
        # print the review id
        print(reviews[review_num]['id'])

        key_phrases = key_phrase_analysis.documents[review_num].key_phrases
        # Print each key phrase
        keyph = []
        for key_phrase in key_phrases:
            keyph.append(key_phrase)
    # Use the client and reviews you created previously to get sentiment scores
    sentiment_analysis = text_analytics_client.sentiment(documents=reviews)
    # Print the results for each review
    for review_num in range(len(reviews)):
        # Get the sentiment score for this review
        sentiment_score = sentiment_analysis.documents[review_num].score
        # classifiy 'positive' if more than 0.5, 
        if sentiment_score < 0.5:
            sentiment = 'negative'
        else:
            sentiment = 'positive'    
        # print file name and sentiment
    return sentiment_score, sentiment, keyph

def listToString(s): 
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele + "," 
    
    # return string  
    return str1 

def brand_detection(url):
    import cv2
    import numpy as np
    import requests
    computervision_client = ComputerVisionClient(cog_endpoint, CognitiveServicesCredentials(cog_key))
    remote_image_url = url
    # Select the visual feature(s) you want
    remote_image_features = ["brands"]
    # Call API with URL and features
    detect_brands_results_remote = computervision_client.analyze_image(remote_image_url, remote_image_features)  
    print("Detecting brands in remote image: ")
    if len(detect_brands_results_remote.brands) == 0:
        print("No brands detected.")
    else:
        for brand in detect_brands_results_remote.brands:
            result = ("'{}' brand detected with confidence {:.1f}% at location {}, {}, {}, {}".format( \
        brand.name, brand.confidence * 100, brand.rectangle.x, brand.rectangle.x + brand.rectangle.w, \
        brand.rectangle.y, brand.rectangle.y + brand.rectangle.h))
    resp = requests.get(remote_image_url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # Blue color in BGR
    color = (255, 0, 0)  
    # Line thickness of 2 px
    thickness = 2
    start_point = (brand.rectangle.x, brand.rectangle.y)
    end_point = ((brand.rectangle.x+brand.rectangle.w),(brand.rectangle.y + brand.rectangle.h))
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    window_name = 'Image'
    thickness = 2
    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.imshow(window_name, image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    return result
    
def text_to_speech(response_text):
    response_text = response_text
    # Configure speech synthesis
    speech_config = SpeechConfig(cog_key, cog_region)
    output_file = os.path.join('data', 'speech', 'response.wav')
    audio_config = AudioOutputConfig(use_default_speaker=True) # Use file instead of default (microphone)
    speech_synthesizer = SpeechSynthesizer(speech_config, audio_config)
    # Transcribe text into speech
    result = speech_synthesizer.speak_text(response_text)
    # Play the output audio file
    IPython.display.display(IPython.display.Audio(audio_config, autoplay=True))
        
        

    
def send():
    while True: ##Loop
        try:
            userInput = ent.get()
            if userInput != "":
                txt.insert(END, "\n"+ "ME: " + ent.get()) #Recieving user input
        except(KeyboardInterrupt, EOFError) as e:
            txt.insert(END,"\n"+"BOT => Bye")
            break
        
        responseAgent ='aiml'
        code = detect_lang(userInput)
        userInput = translate_text(cog_region, cog_key, userInput, to_lang="en", from_lang=code)
        responseAgent ='aiml'
        if responseAgent == 'aiml':
            answer = kern.respond(userInput)
            answer = translate_text(cog_region, cog_key, answer, to_lang=code, from_lang="en")
        #post-process the answer for commands
        if answer[0] == '#':
            params = answer[1:].split('$')
            cmd = int(params[0])
            if cmd == 0: ##If bye is selected
                txt.insert(END,"\n"+" Bot: "+params[1])
                quit()
                break
            elif cmd == 1: ## Webs scraping prices.
                response = requests.get(url_price.format(params[1],params[1])) #adapting users input to the url
                soup= BeautifulSoup(response.text,"lxml") 
                try:
                    price = soup.find_all('div',{'class':'My(6px) Pos(r) smartphone_Mt(6px)'})[0].find('span').text
                    new_answer = "The current price:"
                    new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                    txt.insert(END,"\n"+"BOT => " + new_answer + price)
                except:
                    new_answer = "Sorry, I do not know that. Be more specific!"
                    new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                    txt.insert(END,"\n"+"BOT => " + new_answer)
            elif cmd == 2:
                response = requests.get(url_profile.format(params[1], params[1])) #Web scraping profiles of companies 
                soup = BeautifulSoup(response.text, 'html.parser') 
                pattern = re.compile(r'\s--\sData\s--\s')
                script_data = soup.find('script', text=pattern).contents[0] 
                start = script_data.find("context")-2 
                json_data = json.loads(script_data[start:-12]) #loading date into json
                
                try:
                    new_answer = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['assetProfile']['longBusinessSummary']
                    print(code)
                    new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                    txt.insert(END,"\n"+ "Bot: " + new_answer)  #outputing data to user.
                except KeyError:
                    new_answer = "Sorry, I dont know that stock. Please write the stock initial."
                    new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                    txt.insert(END,"\n"+"Bot: " + new_answer)
                    #engine.say("Sorry, I dont know that stock. Please write the stock initial.") 
                    #engine.runAndWait()
            elif cmd ==3:
                #os.path.isfile('./'+ params[1] + '.PNG') #
                imageName = params[1] ##Conditions to open specific image
                if(imageName == params[1]):
                    new_answer = "Here is a chart of "
                    new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                    txt.insert(END,"\n"+"Bot: "+ new_answer + params[1])
                    openImage("GoogleStockPrices")
                else:
                    new_answer = "Sorry we dont have this chart."
                    new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                    txt.insert(END,"\n"+"Bot: "+ new_answer + params[1]) #error message if image is not found
            elif cmd ==4:
                try:
                    filename1 = browsefile() 
                    cnnModel(filename1, code)
                except:
                    txt.insert(END,"\n"+"Bot: Sorry try again")
                    
            elif cmd ==5: 
                try:
                    imagefileName = params[1]
                    cnnModelChosen(imagefileName)
                except:
                    txt.insert(END,"\n"+"Bot: Sorry try again")
                    
                
            elif cmd ==6:
                path="/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/AI_N0834113(2)/Logos/Predictions/"
                files=os.listdir(path)
                random_file = random.choice(files)
                random_file = path + random_file
                im = Image.open(random_file)
                brandpath = im
                #im.show() #open image
                image_path = random_file
                classNames = cnnModel(image_path, code) #cnn for certain image
                brandNews(classNames, code) #call brand news api for certain clas name
            elif cmd == 7:
                filename1 = browsefile() #open file browser to choose image
                vgg16Model(filename1) #run image through vgg16Model
            elif cmd == 8:
                filename1 = browsefile() #open file browser to choose image 
                ResNet50Model(filename1)#run image through ResNet50
            elif cmd == 9:
                filename1 = browsefile() #open file browser to choose image 
                MobileNet(filename1)#run image through ResNet50
            elif cmd == 31: # if input pattern is "I know that * is *"
                stat_check = True
                try:
                    object,subject=params[1].split((' is a '))
                except:
                    try:
                        object,subject=params[1].split((' is the ceo of '))
                        stat_check = False
                    except:
                        print("error")
                expr=read_expr(subject + '(' + object + ')')
                print(expr)
                expression = subject + '(' + object + ')'
                answer=ResolutionProver().prove(expr, kb, verbose=True)
                print(answer)
                check = False
                for row in data[0]:
                    if expression in row:
                        check = True
                if answer == False:
                    kb.append(expr)
                    if stat_check:
                        txt.insert(END,"\n"+'Bot: OK, I will remember that ' + object + ' is a ' + subject)
                    else:
                        txt.insert(END,"\n"+'OK, I will remember that ' + object +' is the CEO of '+ subject)
                else:
                    if check == True:
                        txt.insert(END,"\n"+"Bot: We already have this.")
                    else:
                        txt.insert(END,"\n"+"Bot: This contradicts our knowledge")
            
            
            # >>> ADD SOME CODES HERE to make sure expr does not contradict 
            # with the KB before appending, otherwise show an error message.
            elif cmd == 33: # if the input pattern is "check that * is *"
                try:
                    object,subject=params[1].split(' is the CEO of ')
                except:
                    try:
                        object,subject=params[1].split(' is ')
                    except:
                        print("error")
                expr=read_expr(subject + '(' + object + ')')
                print(expr)
                answer=ResolutionProver().prove(expr, kb, verbose=True)
                print(answer)
                if answer:
                    new_answer = "That's correct."
                    new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                    txt.insert(END,"\n"+"Bot: " + new_answer)
                else:
                    new_expr = read_expr('-' + subject + '(' + object + ')')
                    print(new_expr)
                    answer=ResolutionProver().prove(new_expr, kb, verbose=True)
                    print(answer)
                    if answer:
                        new_answer = "It may not be true."
                        new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                        txt.insert(END,"\n"+"Bot: " + new_answer)
                    else:
                        new_answer = "This is incorrect"
                        new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                        txt.insert(END,"\n"+"Bot: " + new_answer)
            elif cmd == 34:
                imageName = params[1] ##Conditions to open specific image
                if(imageName == params[1]):
                    new_answer = "Here is a chart of "
                    new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                    txt.insert(END,"\n"+"Bot: "+ new_answer +params[1])
                    new_window = Toplevel(root)
                    # get image
                    image = ImageTk.PhotoImage(Image.open("/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/CNN Dataset/AI_N0834113(2)/IMAGES/"+"GoogleStockPrice.png")) #directory must be changed get correct picture
                    # load image
                    panel = Label(new_window, image=image)
                    panel.image = image
                    panel.pack()
                    
                else:
                    txt.insert(END, "\n" +"Bot: Sorry we dont have this chart.") #error message if image is not found
            elif cmd == 35:#Sentimental analysis and text to speech 
                filename = params[1]
                print(filename)
                sentiment_score, sentiment, keyph = article_review(filename)
                keyph = listToString(keyph)
                new_answer = "The sentiment score of this article is " + str(sentiment_score) 
                new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                txt.insert(END, "\n" +"Bot: " + new_answer)
                #text_to_speech(new_answer)
                new_answer = "The sentiment of this article is " + sentiment
                new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                #text_to_speech(new_answer)
                txt.insert(END, "\n" +"Bot: " + new_answer)
                new_answer = "The key phrases from this article are : "
                new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                txt.insert(END, "\n" +"Bot: " + new_answer)
                new_answer = keyph
                new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                txt.insert(END, "\n" +"Bot: " + new_answer)
                text_to_speech("The sentiment score of this article is " + str(sentiment_score) + "The sentiment of this article is " + sentiment )
            elif cmd == 36: #Command for just sentimental analysis 
                filename = params[1]
                print(filename)
                sentiment_score, sentiment, keyph = article_review(filename)
                keyph = listToString(keyph)
                new_answer = "The sentiment score of this article is " + str(sentiment_score) 
                new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                txt.insert(END, "\n" +"Bot: " + new_answer)
                new_answer = "The sentiment of this article is " + sentiment
                new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                txt.insert(END, "\n" +"Bot: " + new_answer)
                new_answer = "The key phrases from this article are : "
                new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                txt.insert(END, "\n" +"Bot: " + new_answer)
                new_answer = keyph
                new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                txt.insert(END, "\n" +"Bot: " + new_answer)
            elif cmd == 37:#Brand detection 
                import random
                #Generate 5 random numbers between 10 and 30
                n_random = random.randint(0,3)
                urlImages = ["https://cdn.mos.cms.futurecdn.net/uWjEogFLUTBc8mSvagdiuP-970-80.jpg.webp", "https://techcrunch.com/wp-content/uploads/2020/05/GettyImages-1205397116.jpg?w=730&crop=1",
                             "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRe6MDnuTMmnRXA6ArSWbwobsyEGQZ4gh-Ztg&usqp=CAU", "https://images.barrons.com/im-117039?width=620&size=1.5"]
                print(urlImages[n_random])
                result = brand_detection(urlImages[n_random])
                new_answer = result
                new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                txt.insert(END, "\n" +"Bot: " + new_answer)
                
            elif cmd == 99: #Accessing CSV file
                user_response = params[1] 
                user_response = user_response.lower() 
                new_response = gen_response(user_response)
                new_response = new_response.split(",")
                if (new_response != " "):
                    new_answer = new_response[1]
                    new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                    txt.insert(END, "\n" + "Bot:" + new_answer)
                else:
                    new_answer = "Sorry I did not get that, please try again."
                    new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
                    txt.insert(END, "\n" + "Bot: " + new_answer) ##Error message when content is not in CSV file
                sent_tokens.remove(user_response)
        else:
            new_answer = answer
            new_answer = translate_text(cog_region, cog_key, new_answer, to_lang=code, from_lang="en")
            txt.insert(END,"\n Bot: " + new_answer)
            ##engine.say(answer) 
            ##engine.runAndWait()
        ent.delete(0,END) #deletes text in entry box
        
    


##GUI build up
root.resizable(False,False)
txt = Text(root, bd=1, bg='#588B8B', fg='#F28F3B',width = 110, height = 25)
txt.grid(row=0,column=0,columnspan=3)
txt.insert(END,"Welcome to the Technology StockBot. Please feel free to ask questions about technology stocks!")
ent = Entry(root,width=50,fg='black')
send=Button(root,text="Send",command=send).grid(row=1,column=1)
clear=Button(root,text="Clear",command=clear).grid(row=1,column=2)
ent.grid(row=1,column=0)


root.title("TECHNOLOGY STOCK CHATBOT")
root.mainloop()
