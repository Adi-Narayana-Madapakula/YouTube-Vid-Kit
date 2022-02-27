from django.http.response import HttpResponse
from django.shortcuts import render
from django.conf import settings as django_settings
from .models import vidkit as vk
from .models import textAnalysis as ta
from .models import dictSort as ds
from .models import visualImage as vi
import os
from PIL import Image
import glob
from pydub import AudioSegment
# create your views here

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def index(request):
    return render(request, "index.html")

def report(request):
    scene_images= []
    image_numbers=[]
    try:
        try:
            video_id = request.POST['userid']
            video_url = "https://www.youtube.com/watch?v="+str(video_id)
            print("Step - 1")
            video_details = vk.ret_details(video_url)
            print("Step - 2")
            v = vk.save_video(video_url)
            print("Step - 3")
            vid_path = BASE_DIR+"/static/videos/"+str(video_details.title)+"."+str(v)
            print(vid_path)
            print("Step - 4")
            scene_info = vk.find_scenes(vid_path)
            print("Step - 5")
            vk.scene_info(scene_info)
            print(scene_info)
            print("Step - 6")
            vk.extract_audio(video_url)
            audio_file = BASE_DIR + "/static/audios/"+video_details.title+"-"+video_details.videoid+".wav"
            print(audio_file)
            try:
                transcription = vk.extract_text(audio_file)
            except:
                transcription = "Video Does not Have Any Audio. No Transcription can be Made."
            print(transcription)
            print("Step - 8")
            no_of_files = vk.split_audio(audio_file)
            print(no_of_files)
            print("Step - 9")
            try:
                text_data = vk.text_from_split(no_of_files)
                print(text_data)
            except:
                print("Text files not extracted")
            print("Step - 10")
            Text = transcription
            print("Step - 11")
            tokens = Text.split()
            total_length = len(tokens)
            min_length = int((25*total_length)/100)
            max_length = int((40*total_length)/100)
            print("Step - 13")
            #Summary = vk.show_me_summary(Text,min_length,max_length)
            Summary = vk.automatic_summary(Text)
            print("Step - 14")
            Summary = vk.preprocess_summary(Summary)
            #Summary = "My Written Summary"
            print("Step - 15")
            for(root,dirs,files) in os.walk(str(BASE_DIR+"/static/scene_images/")):
                i=0
                for each in files:
                    i=i+1
                    scene_images.append(each)
                    image_numbers.append(i)
            scenes = zip(scene_images,image_numbers)
            print("Step - 16")
            scene_times = vk.scene_times()
            print(scene_times)
            print("Done Successfully Part -1")
            print("Step - 17")
            filterText = ta.textFilter(transcription)
            print("Step - 18")
            tok_freq = ta.textPulse(filterText)
            tok_freq = ds.sortDict(tok_freq)
            print("Step - 19")
            pos_tags = ta.pos_tagging(filterText)
            pos_tags = ds.sortDict(pos_tags)
            print("Step - 20")
            ner_tags = ta.ner_tagging(filterText)
            ner_tags = ds.sortDict(ner_tags)
            print("Done Successfully Part -2")
            vi.visualPOS(pos_tags)
            print("Step -21")
            vi.visualNER(ner_tags)
            print("Step -22")
            #vi.visualPulse(tok_freq)
            print("Step -23")
            vi.visualCloud(filterText)
            print("Step -24")
            print("Done successfully Part -3")
            return render(request, "report.html",{"video_details":video_details, "topwords":tok_freq, "pos_tags":pos_tags, "ner_tags":ner_tags , "times":scene_times, "transcription":Text, "Summary":Summary, "url":str(video_url),"scene_images":scenes, "scene_info":scene_info})
        except:
            try:
                video_url = request.POST['userlink']
                print("Step - 1")
                video_details = vk.ret_details(video_url)
                print("Step - 2")
                v = vk.save_video(video_url)
                print("Step - 3")
                vid_path = BASE_DIR+"/static/videos/"+str(video_details.title)+".mp4"
                print(vid_path)
                print("Step - 4")
                scene_info = vk.find_scenes(vid_path)
                print("Step - 5")
                vk.scene_info(scene_info)
                print(scene_info)
                print("Step - 6")
                vk.extract_audio(video_url)
                audio_file = BASE_DIR + "/static/audios/"+video_details.title+"-"+video_details.videoid+".wav"
                print(audio_file)
                try:
                    transcription = vk.extract_text(audio_file)
                except:
                    transcription = "Video Does not Have Any Audio. No Transcription can be Made."
                print(transcription)
                print("Step - 8")
                no_of_files = vk.split_audio(audio_file)
                print(no_of_files)
                print("Step - 9")
                try:
                    text_data = vk.text_from_split(no_of_files)
                    print(text_data)
                except:
                    print("Text files not extracted")
                print("Step - 10")
                Text = transcription
                print("Step - 11")
                tokens = Text.split()
                total_length = len(tokens)
                print(total_length)
                min_length = int((25*total_length)/100)
                max_length = int((50*total_length)/100)
                print(min_length,max_length)
                print("Step - 13")
                #Summary = vk.show_me_summary(Text,min_length,max_length)
                Summary = vk.automatic_summary(Text)
                print("Step - 14")
                Summary = vk.preprocess_summary(Summary)
                #Summary = "My Written Summary"
                print("Step - 15")
                for(root,dirs,files) in os.walk(str(BASE_DIR+"/static/scene_images/")):
                    i=0
                    for each in files:
                        i=i+1
                        scene_images.append(each)
                        image_numbers.append(i)
                scenes = zip(scene_images,image_numbers)
                print("Step - 16")
                scene_times = vk.scene_times()
                print(scene_times)
                print("Done Successfully Part -1")
                print("Step - 17")
                filterText = ta.textFilter(transcription)
                print("Step - 18")
                tok_freq = ta.textPulse(filterText)
                tok_freq = ds.sortDict(tok_freq)
                print("Step - 19")
                pos_tags = ta.pos_tagging(filterText)
                pos_tags = ds.sortDict(pos_tags)
                print("Step - 20")
                ner_tags = ta.ner_tagging(filterText)
                ner_tags = ds.sortDict(ner_tags)
                print("Done Successfully Part -2")
                vi.visualPOS(pos_tags)
                print("Step -21")
                vi.visualNER(ner_tags)
                print("Step -22")
                #vi.visualPulse(tok_freq)
                print("Step -23")
                vi.visualCloud(filterText)
                print("Step -24")
                print("Done successfully Part -3")
                return render(request, "report.html",{"video_details":video_details, "topwords":tok_freq, "pos_tags":pos_tags, "ner_tags":ner_tags ,"times":scene_times, "transcription":Text, "Summary":Summary, "url":str(video_url),"scene_images":scenes, "scene_info":scene_info})
            except:
                return render(request, "index.html")
    except:
        return render(request, "index.html")   

def feedback(request):
    
    return render(request, "feedback.html")

def contact(request):
    return render(request, "contact.html")