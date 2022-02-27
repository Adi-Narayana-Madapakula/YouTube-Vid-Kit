from __future__ import unicode_literals
from django.db import models
from django.conf import settings as django_settings

'''
'''
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS  
import numpy as np
from PIL import Image
from urllib.parse import urlparse, parse_qs
import pafy
import operator

from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

# Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect import scene_manager
#from IPython.display import YouTubeVideo
#from IPython.display import Audio,Image

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

from pytube import YouTube
from pydub import AudioSegment
from pydub import AudioSegment
import speech_recognition as sr
from pydub.silence import split_on_silence

from transformers import pipeline
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

import seaborn as sns
import pandas as pd
sns.set_style('darkgrid')
import matplotlib.pyplot as plt

import os, shutil

from fpdf import FPDF
import numpy as np
import cv2

import spacy

import youtube_dl

import warnings
warnings.filterwarnings('ignore')
'''
'''

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class visualImage():
    def visualPOS(d):
        print("Start POS Image")
        labels=[]
        count=[]
        print("Step-101")
        for key, value in d.items():
            labels.append(key)
            count.append(value[0][0])
        print("Step-102")
        df=pd.DataFrame({'label':labels,'no':count})
        print("Step-103")
        ax=sns.barplot(x='label',y=count,data=df)
        print("Step-104")
        for p in ax.patches:
            ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)
        print("Step-105")
        plt.xlabel("POS Tags")
        plt.ylabel("Frequency")
        plt.title("POS Tags Frequency Distribution")
        save_path = BASE_DIR + "/static/visualisations/"
        print("End POS Image ")
        os.chdir(save_path)
        plt.savefig("pos.jpg")
        os.chdir(BASE_DIR)
        print("POS Image Saved")
    
    def visualNER(d):
        print("Start NER Image")
        labels=[]
        count=[]
        for key, value in d.items():
            labels.append(key)
            count.append(value[0][0])
        df=pd.DataFrame({'label':labels,'no':count})
        ax=sns.barplot(x='label',y=count,data=df)

        for p in ax.patches:
            ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)
        plt.xlabel("NER Tags")
        plt.ylabel("Frequency")
        plt.title("NER Tags Frequency Distribution")
        save_path = BASE_DIR + "/static/visualisations/"
        print("End NER Image ")
        os.chdir(save_path)
        plt.savefig("ner.jpg")
        os.chdir(BASE_DIR)
        print("NER Image Saved")

    def visualPulse(d):
        labels=[]
        count=[]
        for key, value in d.items():
            labels.append(key)
            count.append(value)
        df=pd.DataFrame({'label':labels,'no':count})
        ax=sns.barplot(x='label',y=count,data=df)

        for p in ax.patches:
            ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)
        plt.xlabel("KeyWords")
        plt.ylabel("Frequency")
        plt.title("Keyword Frequency Distribution")
        save_path = BASE_DIR + "/static/visualisations/"
        print("End Pulse Image ")
        os.chdir(save_path)
        plt.savefig("pulse.jpg")
        os.chdir(BASE_DIR)
        print("Pulse Image Saved")

    def visualCloud(Text):
        print("Start CLoud Image")
        #utube = BASE_DIR + "/static/assets/images/rguktlogo.jpg"
        #mask = np.array(Image.open(utube))
        wc = WordCloud(stopwords = STOPWORDS, 
               background_color = "white",
               max_words = 2000,
               max_font_size = 500,
               random_state = 42)
        wc.generate(Text)
        plt.title("WordCloud for your Video")
        save_path = BASE_DIR + "/static/visualisations/"
        print("End Cloud Image")
        os.chdir(save_path)
        plt.savefig("cloud.jpg")
        os.chdir(BASE_DIR)
        print("CLoud Image Saved")

    

class dictSort():
    def sortDict(d, reverse = True):
        return dict(sorted(d.items(), key = lambda x: x[1], reverse = reverse))

class PDF(FPDF):
  w = 210
  h = 297/2
  def image_and_text(self, image_name, text):
      w = 210
      h = 297/2
      #self.add_page()
      self.set_xy(20, 2)
      self.image(image_name, type='JPG', w = 210, h = 297/2)
      self.set_xy(2, h+4)
      self.set_font('Arial', 'B', 12)
      self.set_text_color(0, 0, 0)
      #self.cell(w=40, h=40.0, align='C', txt=text, border=0)
      self.multi_cell(290,10,text)

class vidkit:
    def run_command(command):
        print("Command Execution Started")
        os.system((command))
        print("Command Executed Successfully")

    def ret_details(video_url):
        print("Pafy-1")
        video_details = pafy.new(video_url)
        print("Pafy-2")
        return video_details

    def save_video(URL):
        # URL = qX8RX1XalEg
        video=pafy.new(URL)
        #best=video.getbest()
        print("video best")
        best_video = video.getbest(preftype="mp4")
        video_dir = BASE_DIR +"/static/videos/"
        print(video_dir)
        try:
            print("Enter Video Save")
            try:
                v=best_video.download(str(video_dir))
            except:
                print("Didnt get best")
            print("Video Save Gone")          
        except:
            print("Video Not Saved")
        return best_video.extension
    
    def find_scenes(video_path, threshold=30.0):
        # Create our video & scene managers, then add the detector.
        print("Step1")
        video_manager = VideoManager([video_path])
        print("Step2")
        sm = SceneManager()
        print("Step3")
        sm.add_detector(ContentDetector(threshold=30.0))

        # Improve processing speed by downscaling before processing.
        video_manager.set_downscale_factor()

        # Start the video manager and perform the scene detection.
        video_manager.start()
        sm.detect_scenes(frame_source=video_manager)
        scenes = sm.get_scene_list()
        folder_name = BASE_DIR + "/static/scene_images/"
        # create a directory to store the images
        print("Directory Checking")
        if not os.path.isdir(folder_name):
            print("Directory not found")
            os.mkdir(folder_name)
        else:
            print("Directory Found")
            for f in os.listdir(folder_name):
                print("Removing files")
                os.remove(os.path.join(folder_name, f))
            print("Files Removed")
        output_directory = BASE_DIR + "/static/scene_images/"
        print("Output Directory",output_directory)
        scene_manager.save_images(scenes, video_manager, num_images=1, frame_margin=1, image_extension='jpg', encoder_param=95, image_name_template='$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER', output_dir=output_directory, downscale_factor=1, show_progress=True)
        # Each returned scene is a tuple of the (start, end) timecode.
        return scenes
    
    def scene_info(scenes):
        location = BASE_DIR + "/static/scene-info.txt"
        temp_writer = open(location,'w')
        for i in range(len(scenes)):
            temp_writer.write(str(scenes[i][0])+" "+str(scenes[i][1])+" "+"part-0"+str(i+1)+".wav"+"\n")
        temp_writer.close()


    def save_audio(URL):
        # URL = qX8RX1XalEg
        audio=pafy.new(URL)
        #best=video.getbest()
        print("Audio best")
        best_audio = audio.getbestaudio()
        audio_dir = BASE_DIR +"/static/audios/"
        print(audio_dir)
        try:
            print("Enter Audio Save")
            try:
                v=best_audio.download(str(audio_dir))
            except:
                print("Didnt get best")
            print("Audio Save Gone")          
        except:
            print("Audio Not Saved")
        return best_audio.extension

    def extract_audio(URL):
        ydl_opts = {
            'format' : 'bestaudio/best',
            'postprocessors' : [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec':'wav',
            }],
            'postprocessor_args':[
                '-ar','16000'
            ],
            'prefer_ffmpeg': True,
            'keepvideo' : True
        }
        audio_directory = BASE_DIR + "/static/audios/"
        os.chdir(audio_directory)
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([str(URL)])
        os.chdir(BASE_DIR)

    def extract_text(audio_file):
        print("Text -1")
        r = sr.Recognizer()
        print("Text -2")
        sound = AudioSegment.from_wav(str(audio_file))
        print("Text -3")
        # split audio sound where silence is 700 miliseconds or more and get chunks
        chunks = split_on_silence(sound,
            # experiment with this value for your target audio file
            min_silence_len = 1000,
            # adjust this per requirement
            silence_thresh = sound.dBFS-14,
            # keep the silence for 1 second, adjustable as well
            keep_silence=400,
        )
        print("Text -4")
        folder_name = BASE_DIR+"/static/audio-chunks"
        # create a directory to store the audio chunks
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        else:
            for f in os.listdir(folder_name):
                os.remove(os.path.join(folder_name, f))
        print("Text -5")
        whole_text = ""
        # process each chunk 
        print("Text -6")
        print(chunks)
        for i, audio_chunk in enumerate(chunks, start=1):
            # export audio chunk and save it in
            # the `folder_name` directory.
            print("Text -7")
            chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
            print("Text -8")
            audio_chunk.export(chunk_filename, format="wav")
            # recognize the chunk
            print("Text -9")
            with sr.AudioFile(chunk_filename) as source:
                print("Text -10")
                audio_listened = r.record(source)
                print("Text -11")
                # try converting it to text
                try:
                    print("Text -12")
                    text = r.recognize_google(audio_listened)
                    print("Text -13")
                except sr.UnknownValueError as e:
                    print("Text -14")
                    print("Error:", str(e))
                else:
                    print("Text -15")
                    text = f"{text.capitalize()}. "
                    print("Text -16")
                    print(chunk_filename, ":", text)
                    print("Text -17")
                    whole_text += text
        # return the text for all chunks detected
        return whole_text

    def split_audio(audio_file):
        with open(str(BASE_DIR+'/static/scene-info.txt'), 'r') as f:
            i=0
            scene_times = []
            for line in f:
                if line.startswith('#') or len(line) < 1:
                    continue
                start, end, name = line.strip().split() #00:00:00.000
                start_hours, start_mins, start_secs = start.split(":")
                start_secs, start_millis = start_secs.split(".")
                end_hours, end_mins, end_secs = end.split(":")
                end_secs, end_millis = end_secs.split(".")
                start_trim = int((start_hours*3600000)+(60000*start_mins)+(1000*start_secs)+start_millis)
                end_trim = int((end_hours*3600000)+(60000*end_mins)+(1000*end_secs)+end_millis)
                newAudio = AudioSegment.from_wav(audio_file)
                newAudio = newAudio[start_trim:end_trim]
                #scene_times.append([start_trim,end_trim])
                os.chdir(str(BASE_DIR+'/static/audio_scenes/'))
                newAudio.export(name,format="wav")
                i=i+1
                os.chdir(BASE_DIR)
            return i


    def text_from_split(self, no_of_files):
        print("Started Text - 1")
        text_data = []
        filename = BASE_DIR + "/static/audio_scenes/part-0{num}.wav"
        for i in range(no_of_files):
            print("Started Text - 2")
            audio_file = filename.format(i+1)
            print("Started Text - 3")
            text_data.append(self.extract_text(audio_file))
            print("Started Text - 4")
        return text_data

    def show_me_summary(transcription,min,max): 
        print("Step - 1")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        print("Step - 2")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        print("Step - 3")
        summary_text = tokenizer.encode(transcription, return_tensors="pt", truncation=True)
        print("Step - 4")
        outputs = model.generate(
            summary_text, 
            max_length=max, 
            min_length=min,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True)
        return tokenizer.decode(outputs[0])

    def automatic_summary(transcription):
        auto_abstractor = AutoAbstractor()
        auto_abstractor.tokenizable_doc = SimpleTokenizer()
        auto_abstractor.delimiter_list = [".", "\n"]
        abstractable_doc = TopNRankAbstractor()
        summary_text = ""
        result_dict = auto_abstractor.summarize(transcription, abstractable_doc)
        for sentence in result_dict["summarize_result"]:
            summary_text = summary_text +"\n"+sentence
        return summary_text
        
    
    def preprocess_summary(summary):
        summary = summary.replace('<pad>', '')
        summary = summary.replace('</s>', '')
        #summary = summary[1:]
        #print(summary)
        #summary.capitalize()
        #print(summary)
        return summary
    
    def getSceneImages(title,n):
        image_name = "{title}-Scene-00{scene_number}-0{image_number}.jpg"
        image_list = []
        for i in range(n):
            if i<=9:
                image_list.append(image_name.format(title,i+1,i+1))
            else:
                image_list.append("{title}-Scene-0{scene_number}-{image_number}.jpg".format(title,i+1,i+1))
        return image_list

    def scene_times():
        print("Times-1")
        location = BASE_DIR + "/static/scene-info.txt"
        print(location)
        temp_reader = open(location,'r')
        print("Times-3")
        lines = temp_reader.readlines()
        print("Times-4")
        final_times = []
        for line in lines:
            start, end, name = line.split()
            final_times.append([start.split(":"),end.split(":")])
        temp_reader.close()
        print("Times Done")
        return final_times

class textAnalysis():
    def textFilter(Text):
        print("textFilter Starts")
        nlp = spacy.load('en_core_web_sm')
        print("filter - 1")
        stopwords = spacy.lang.en.stop_words.STOP_WORDS
        print("filter - 2")
        filterText = Text.lower()
        print("filter - 3")
        doc = nlp(filterText, disable=['ner', 'parser'])
        print("filter - 4")
        lemmas = [token.lemma_ for token in doc]
        print("filter - 5")
        a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma not in stopwords]
        print("textFilter Ends")
        return ' '.join(a_lemmas)

    def textPulse(Text):
        print("textPulse Starts")
        textPulse = {}
        doc=[ each.strip().replace(".","") for each in Text.split() ]
        i=0
        for token in doc:       
            if i>=15:
                break     
            if token not in textPulse:
                textPulse[token]=1
                i=i+1
            else:
                textPulse[token]=textPulse[token]+1
                i=i+1
        print("textPulse Ends")
        return textPulse

    def pos_tagging(Text):
        print("POS Starts")
        nlp = spacy.load('en_core_web_sm')
        pos_dict = {}
        final_pos = {}
        doc = nlp(Text)
        pos = [[token.text, token.pos_] for token in doc]
        #print(pos[0],pos[1])
        for i in range(len(pos)):
            if pos[i][1] not in pos_dict:
                pos_dict[pos[i][1]] = []
            pos_dict[pos[i][1]].append(pos[i][0])

        #print(pos_dict)
        for tag in pos_dict:

            if tag not in final_pos:
                final_pos[tag] = []
                tag_freq = len(set(pos_dict[tag]))
                tag_list = list(set(pos_dict[tag]))
                final_pos[tag].append(tuple([tag_freq,tag_list]))
            else:
                continue
        #print(final_pos)
        print("POS Ends")
        return final_pos

    def ner_tagging(Text):
        print("NER Starts")
        nlp = spacy.load('en_core_web_sm')
        ner_dict = {}
        final_ner = {}
        doc = nlp(Text)
        ner = [[ent.text,ent.label_] for ent in doc.ents ]
        for i in range(len(ner)):
            if ner[i][1] not in ner_dict:
                ner_dict[ner[i][1]] = []
            ner_dict[ner[i][1]].append(ner[i][0])
        #print(ner_dict)
        for tag in ner_dict:
            if tag not in final_ner:
                final_ner[tag] = []
                tag_freq = len(set(ner_dict[tag]))
                tag_list = list(set(ner_dict[tag]))
                final_ner[tag].append(tuple([tag_freq,tag_list]))
            else:
                continue
        #print(final_ner)
        print("NER Ends")
        return final_ner