''' IN THIS EXAMPLE WE WILL DOWNLOAD 10 FIRST IMAGES FOR EACH EXISTING POKEMON.
    GOOGLE FIRST IMAGES ARE REALLY COOL USUALLY IF YOU FIND THE GOOD WORDS TO LOOK FOR,
    SO WE DO NOT NEED EVEN THE CNN CLASSIFIER PART. IT'S NOT IDEAL BUT IT REALLY WORKS '''

from GoogleImageSpider import *
from ImageNormalizer import *
import certifi, urllib3
import time
import json
import os

path = "pokemon_images"
images_per_pokemon = 10
start_time = time.time()

def print_time():
    # TIMING CONTROL
    elapsed_time = time.time() - start_time
    mins = int(elapsed_time / 60)
    secs = elapsed_time - (mins * 60)
    print("Accumulative time: %02d:%02d" % (mins, int(secs % 60)))

if not os.path.exists(path):
    os.makedirs(path)

# FIRST USE A POKEMON API TO GET EACH POKEMON NAME:
http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
json_string = http.request('GET', 'https://pokeapi.co/api/v2/pokemon/?limit=1000')
json_string = json_string.data.decode('utf-8')
pokemons = json.loads(json_string)
print("Pokemons in API: ", len(pokemons['results']))
print_time()

# SECOND STEP DOWNLOAD IMAGES FROM EACH POKEMON NAME:
gis = GoogleImageSpider()
for poketemp in pokemons['results']:
    gis.get_images("pokemon " + poketemp['name'] + " image", images_per_pokemon)
    gis.save_images(poketemp['name'], path)
    gis.clear()
    print (poketemp['name'], "downloaded")
    print_time()

print ("All pokemon images downloaded")
print_time()

# THIRD STEP NORMALIZE IMAGES
im = ImageNormalizer(path)
im.normalize((256, 256), "destin_images")
print_time()
print ("Done!")
