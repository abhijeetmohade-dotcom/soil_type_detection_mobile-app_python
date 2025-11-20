import requests

image_path = "test_soil.jpg"  # Make sure this file exists in D:\Projects\testtyy

url = 'http://127.0.0.1:5001/predict'
files = {'image': open(image_path, 'rb')}
response = requests.post(url, files=files)
print(response.json())
