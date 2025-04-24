# system
import os
import sys
import shutil
import datetime
import queue
import threading
import traceback
import subprocess
import string 
import random 
import json 
import time 
import logging
import warnings
import yaml
import copy
import re
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Tuple, Union,Optional
from pathlib import Path

# machine learn
import numpy as np
import onnxruntime
import faster_whisper
import jieba 
import sklearn
import scipy

# audio
FFMPEG = "ffmpeg"
import soundfile as sf 
import kaldi_native_fbank as knf

# web
import ssl 
import asyncio
import websockets
import requests