#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:16:49 2020

@author: RishiSingh
"""
##Libraries used
from tkinter import *
import re
import json
import csv
import pyttsx3
from io import StringIO
from bs4 import BeautifulSoup
import requests
import os
import csv
import nltk
import aiml
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image,ImageTk
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
# Main loop
#######################################################

url_profile = "https://finance.yahoo.com/quote/{}/profile?p={}" ##URL for web scrapping
url_price = 'https://uk.finance.yahoo.com/quote/{}?p={}'
# engine = pyttsx3.init()  
# voices = engine.getProperty('voices') 
# engine.setProperty('voice', voices[0].id)
#Text to speach does not work fully when using GUI but it does when using console.
file = open('TechQA.csv','r',errors = 'ignore') # Data from CSV file.

corpus = file.read()
corpus = corpus.lower()
nltk.download('punkt') 
nltk.download('wordnet')
##Preprocessing data
sent_tokens = nltk.sent_tokenize(corpus)# converts to list of sentences 
word_tokens = nltk.word_tokenize(corpus)# converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()

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
    
def open(imageName): #Opens image
    new_window = Toplevel(root)
    # get image
    image = ImageTk.PhotoImage(Image.open("/Users/RishiSingh/Computer Science University /Third Year - 2020-2021/Artificial Intelligence/Chatbot/IMAGES/"+imageName+".png")) #directory must be changed get correct picture
    # load image
    panel = Label(new_window, image=image)
    panel.image = image
    panel.pack()
    
def quit(): #To quit GUI
    root.destroy()
    
    
    
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
        
        if responseAgent == 'aiml':
            answer = kern.respond(userInput)
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
                    txt.insert(END,"\n"+"BOT => The current price: " + price)
                except:
                    txt.insert(END,"\n"+"BOT => Sorry, I do not know that. Be more specific!")
            elif cmd == 2:
                response = requests.get(url_profile.format(params[1], params[1])) #Web scraping profiles of companies 
                soup = BeautifulSoup(response.text, 'html.parser') 
                pattern = re.compile(r'\s--\sData\s--\s')
                script_data = soup.find('script', text=pattern).contents[0] 
                start = script_data.find("context")-2 
                json_data = json.loads(script_data[start:-12]) #loading date into json
                try:
                    txt.insert(END,"\n"+ "Bot: " +json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['assetProfile']['longBusinessSummary'])  #outputing data to user.
                except KeyError:
                    txt.insert(END,"\n"+"Bot: Sorry, I dont know that stock. Please write the stock initial.")
                    #engine.say("Sorry, I dont know that stock. Please write the stock initial.") 
                    #engine.runAndWait()
            elif cmd ==3:
                #os.path.isfile('./'+ params[1] + '.PNG') #
                imageName = params[1] ##Conditions to open specific image
                if(imageName == params[1]):
                    txt.insert(END,"\n"+"Bot: Here is a chart of "+params[1])
                    open(imageName)
                    
                else:
                    txt.insert(END, "\n" +"Bot: Sorry we dont have this chart.") #error message if image is not found
            elif cmd == 99: #Accessing CSV file
                user_response = params[1] 
                user_response = user_response.lower() 
                new_response = gen_response(user_response)
                if (new_response != " "):
                    txt.insert(END, "\n" + "Bot:" +new_response)
                else:
                    txt.insert(END, "\n" + "Bot: Sorry I did not get that, please try again.") ##Error message when content is not in CSV file
                sent_tokens.remove(user_response)
        else: 
            txt.insert(END,"\n Bot: " + answer)
            ##engine.say(answer) 
            ##engine.runAndWait()
        ent.delete(0,END) #deletes text in entry box

##GUI build up
root.resizable(False,False)
txt = Text(root, bd=1, bg='black', fg='green',width = 100, height = 25)
txt.grid(row=0,column=0,columnspan=3)
txt.insert(END,"Welcome to the Technology StockBot. Please feel free to ask questions about technology stocks!")
ent = Entry(root,width=50,fg='black',)
send=Button(root,text="Send",command=send).grid(row=1,column=1)
clear=Button(root,text="Clear",command=clear).grid(row=1,column=2)
ent.grid(row=1,column=0)
root.title("TECHNOLOGY STOCK CHATBOT")
root.mainloop()