import requests

def download_from_google_drive(id, fname, file):
    url = f'https://drive.google.com/uc?export=download&id={id}'
    