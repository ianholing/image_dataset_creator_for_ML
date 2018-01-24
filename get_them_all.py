from GoogleImageSpider import *
from ImageNormalizer import *
import certifi, urllib3
import json
import os

path = "pokemon_images"
images_to_download = 100

if not os.path.exists(path):
    os.makedirs(path)

# FIRST USE A POKEMON API TO GET EACH POKEMON NAME:
http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
json_string = http.request('GET', 'https://pokeapi.co/api/v2/pokemon/?limit=1000')
json_string = json_string.data.decode('utf-8')
pokemons = json.loads(json_string)
print("Pokemons in API: ", len(pokemons['results']))

# SECOND STEP DOWNLOAD IMAGES FROM EACH POKEMON NAME:
gis = GoogleImageSpider()
for poketemp in pokemons['results']:
    gis.get_images("pokemon " + poketemp['name'] + " image", images_to_download)
    gis.save_images(poketemp['name'], path)
    gis.clear()
    print (poketemp['name'], "downloaded")

print ("All pokemon images downloaded")

# THIRD STEP NORMALIZE IMAGES
im = ImageNormalizer(path)
im.normalize((256, 256), "destin_images")
print ("Done!")