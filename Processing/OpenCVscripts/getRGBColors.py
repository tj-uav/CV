import requests

request = requests.get('http://chir.ag/projects/name-that-color/#3C7890')
text = request.text
print(text)