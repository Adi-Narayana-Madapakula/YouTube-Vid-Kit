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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def index(request):
    return render(request, "index.html")

def report(request):
    scene_images = []
    image_numbers = []

    def process_video(video_url):
        video_details = vk.ret_details(video_url)
        v = vk.save_video(video_url)
        vid_path = f"{BASE_DIR}/static/videos/{video_details.title}.{v}"
        scene_info = vk.find_scenes(vid_path)
        vk.scene_info(scene_info)
        vk.extract_audio(video_url)
        audio_file = f"{BASE_DIR}/static/audios/{video_details.title}-{video_details.videoid}.wav"
        try:
            transcription = vk.extract_text(audio_file)
        except:
            transcription = "Video Does not Have Any Audio. No Transcription can be Made."
        no_of_files = vk.split_audio(audio_file)
        try:
            text_data = vk.text_from_split(no_of_files)
        except:
            text_data = None
        Text = transcription
        tokens = Text.split()
        total_length = len(tokens)
        min_length = int((25 * total_length) / 100)
        max_length = int((40 * total_length) / 100)
        Summary = vk.automatic_summary(Text)
        Summary = vk.preprocess_summary(Summary)
        scene_times = vk.scene_times()
        filterText = ta.textFilter(transcription)
        tok_freq = ta.textPulse(filterText)
        tok_freq = ds.sortDict(tok_freq)
        pos_tags = ta.pos_tagging(filterText)
        pos_tags = ds.sortDict(pos_tags)
        ner_tags = ta.ner_tagging(filterText)
        ner_tags = ds.sortDict(ner_tags)
        vi.visualPOS(pos_tags)
        vi.visualNER(ner_tags)
        vi.visualCloud(filterText)
        return {
            "video_details": video_details,
            "topwords": tok_freq,
            "pos_tags": pos_tags,
            "ner_tags": ner_tags,
            "times": scene_times,
            "transcription": Text,
            "Summary": Summary,
            "scene_images": scene_images,
            "scene_info": scene_info
        }

    if request.method == 'POST':
        try:
            video_id = request.POST.get('userid')
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            return render(request, "report.html", process_video(video_url))
        except:
            try:
                video_url = request.POST.get('userlink')
                return render(request, "report.html", process_video(video_url))
            except:
                return render(request, "index.html")
    return render(request, "index.html")

def feedback(request):
    return render(request, "feedback.html")

def contact(request):
    return render(request, "contact.html")
