from GoogleImageSpider import *
from ImageNormalizer import *
import certifi, urllib3
import json
import os

path = "pokemon_images"
if not os.path.exists(path):
    os.makedirs(path)

http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
json_string = http.request('GET', 'https://pokeapi.co/api/v2/pokemon/?limit=1000')
json_string = json_string.data.decode('utf-8')
pokemons = json.loads(json_string)

print("Pokemons in API: ", len(pokemons['results']))
gis = GoogleImageSpider()
im = ImageNormalizer(path)
for poketemp in pokemons['results']:
    gis.get_images("pokemon " + poketemp['name'] + " image", 50)
    gis.save_images(poketemp['name'], path)
    gis.clear()

print ("Done!")