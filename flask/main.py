import io
import re
import mlflow
import pickle
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from flask_cors import CORS
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.dates as mdates
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MIflowClient
from flask import Flask, request, jsonify, send_file


# Initilize the Flask
app = Flask(__name__)
CORS(app)

