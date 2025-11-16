import cv2
import requests
from streamlink import Streamlink
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import psycopg2
import numpy as np
import time
import os
import threading
from sklearn.metrics.pairwise import cosine_similarity
import json




def get_stream_url(channel):
    session = Streamlink()
    streams = session.streams(f'https://www.twitch.tv/{channel}')
    return streams['best'].url if streams else None

channelname = "jasontheween"
print(get_stream_url(channelname))