"""
A standalone script to download and parse edgar 10k MDA section
"""
import argparse
import csv
import concurrent.futures
import itertools
import os
import time
import re
import unicodedata
from collections import namedtuple
from functools import wraps
from glob import glob
from pathlib import Path
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup 
import subprocess
import pandas as pd

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
    
def create_parser():
    """Argument Parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str,
                        default="./data", help="path to save data")
    parser.add_argument('--overwrite', action="store_true",
                        help="If True, overwrites downloads and processed files.")
    parser.add_argument('--debug', action="store_true",
                        help="Debug mode")
    return parser


def main():
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()

    form_dir = Path("./data/form10k")

    # Normalize forms
    parsed_form_dir = os.path.join(args.data_dir, "form10k.parsed")
    parse_html_multiprocess(form_dir, parsed_form_dir, args.overwrite)

    # Parse MDA
    mda_dir = os.path.join(args.data_dir, "mda")
    parse_mda_multiprocess(parsed_form_dir, mda_dir, args.overwrite)

    subprocess.run(["python", "visualizer.py"])


def write_content(content, output_path):
    """ Writes content to file
    Args:
        content (str)
        output_path (str): path to output file
    """
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.write(content)


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kw):
        start_time = time.time()
        result = f(*args, **kw)
        end_time = time.time()
        print("{} took {:.2f} seconds."
              .format(f.__name__, end_time-start_time))
        return result
    return wrapper

def read_url_from_combined_csv(csv_path):
    """ Reads url from csv file
    Args:
        csv_path (str): path to index file
    Returns
        urls: urls in combined csv
    """
    urls = []
    with open(csv_path, 'r') as fin:
        reader = csv.reader(fin, delimiter=",",
                            quotechar='\"', quoting=csv.QUOTE_ALL)
        # Skip header
        next(reader)
        for row in reader:
            url = row[-1]
            urls.append(url)
    return urls


def parse_html_multiprocess(form_dir, parsed_form_dir, overwrite=True):
    """ parse html with multiprocess
    Args:
        form_dir (str)
    Returns:
        parsed_form_dir (str)
    """
    # Create directory
    os.makedirs(parsed_form_dir, exist_ok=True)

    # Prepare argument
    form_paths = sorted(glob(os.path.join(form_dir, "*.txt")))
    parsed_form_paths = []
    for form_path in form_paths:
        form_name = os.path.basename(form_path)
        parsed_form_path = os.path.join(parsed_form_dir, form_name)
        parsed_form_paths.append(parsed_form_path)

    # Multiprocess
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for form_path, parsed_form_path in zip(form_paths, parsed_form_paths):
            executor.submit(parse_html,
                            form_path, parsed_form_path, overwrite)


def parse_html(input_file, output_file, overwrite=True):
    """ Parses text from html with BeautifulSoup
    Args:
        input_file (str)
        output_file (str)
    """
    if not overwrite and os.path.exists(output_file):
        print("{} already exists.  Skipping parse html...".format(output_file))
        return

    print("Parsing html {}".format(input_file))
    with open(input_file, 'r') as fin:
        content = fin.read()
    # Parse html with BeautifulSoup
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text("\n")
    write_content(text, output_file)
    # Log message
    print("Write to {}".format(output_file))


def normalize_text(text):
    """Normalize Text
    """
    text = unicodedata.normalize("NFKD", text)  # Normalize
    text = '\n'.join(text.splitlines())  # Unicode break lines

    # Convert to upper
    text = text.upper()  # Convert to upper

    # Take care of breaklines & whitespaces combinations due to beautifulsoup parsing
    text = re.sub(r'[ ]+\n', '\n', text)
    text = re.sub(r'\n[ ]+', '\n', text)
    text = re.sub(r'\n+', '\n', text)

    # To find MDA section, reformat item headers
    text = text.replace('\n.\n', '.\n')  # Move Period to beginning

    text = text.replace('\nI\nTEM', '\nITEM')
    text = text.replace('\nITEM\n', '\nITEM ')
    text = text.replace('\nITEM  ', '\nITEM ')

    text = text.replace(':\n', '.\n')

    # Math symbols for clearer looks
    text = text.replace('$\n', '$')
    text = text.replace('\n%', '%')

    # Reformat
    text = text.replace('\n', '\n\n')  # Reformat by additional breakline

    return text


def parse_mda_multiprocess(form_dir: str, mda_dir: str, overwrite: bool = False):
    """ Parse MDA section from forms with multiprocess
    Args:
        form_dir (str)
        mda_dir (str)
    """
    # Create output directory
    os.makedirs(mda_dir, exist_ok=True)

    # Prepare arguments
    form_paths = sorted(glob(os.path.join(form_dir, "*")))
    mda_paths = []
    for form_path in form_paths:
        form_name = os.path.basename(form_path)
        root, _ = os.path.splitext(form_name)
        mda_path = os.path.join(mda_dir, '{}.mda'.format(root))
        mda_paths.append(mda_path)

    # Multiprocess
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for form_path, mda_path in zip(form_paths, mda_paths):
            executor.submit(parse_mda, form_path, mda_path, overwrite)


def parse_mda(form_path, mda_path, overwrite=False):
    """ Reads form and parses mda
    Args:
        form_path (str)
        mda_path (str)
    """
    if not overwrite and os.path.exists(mda_path):
        print("{} already exists.  Skipping parse mda...".format(mda_path))
        return

    # Read
    print("Parse MDA {}".format(form_path))
    with open(form_path, "r") as fin:
        text = fin.read()

    # Normalize text here
    text = normalize_text(text)

    # Parse MDA
    mda, end = find_mda_from_text(text)
    # Parse second time if first parse results in index
    if mda and len(mda.encode('utf-8')) < 1000:
        mda, _ = find_mda_from_text(text, start=end)

    if mda:
        print("Write MDA to {}".format(mda_path))
        mda = remove_irrelevancy(mda)
        output = query({
                        "inputs": mda,
                        "min_length" : 200
                        })
        write_content(output[0]['summary_text'], mda_path)
    else:
        print("Parse MDA failed {}".format(form_path))


def find_mda_from_text(text, start=0):
    """Find MDA section from normalized text
    Args:
        text (str)s
    """
    debug = False

    mda = ""
    end = 0

    # Define start & end signal for parsing
    item7_begins = [
        '\nITEM 7.', '\nITEM 7 –', '\nITEM 7:', '\nITEM 7 ', '\nITEM 7\n'
    ]
    item7_ends = ['\nITEM 7A']
    if start != 0:
        item7_ends.append('\nITEM 7')  # Case: ITEM 7A does not exist
    item8_begins = ['\nITEM 8']
    """
    Parsing code section
    """
    text = text[start:]

    # Get begin
    for item7 in item7_begins:
        begin = text.find(item7)
        if debug:
            print(item7, begin)
        if begin != -1:
            break

    if begin != -1:  # Begin found
        for item7A in item7_ends:
            end = text.find(item7A, begin + 1)
            if debug:
                print(item7A, end)
            if end != -1:
                break

        if end == -1:  # ITEM 7A does not exist
            for item8 in item8_begins:
                end = text.find(item8, begin + 1)
                if debug:
                    print(item8, end)
                if end != -1:
                    break

        # Get MDA
        if end > begin:
            mda = text[begin:end].strip()
        else:
            end = 0

    return mda, end

def remove_irrelevancy(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)
    
    # Remove non-alphanumeric characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text



if __name__ == "__main__":
    main()