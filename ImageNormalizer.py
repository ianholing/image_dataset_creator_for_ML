from PIL import Image
import sys, os

class ImageNormalizer:
    origin_path = None
    debug = False
    
    def __init__(self, origin_path, debug=False):
        self.origin_path = origin_path
        self.debug = debug
        
    def normalize(self, size, destin_path, fill_color=(255, 255, 255), delete_originals=False):
        print ("Normalizing..")
        
        if not os.path.exists(destin_path):
            os.makedirs(destin_path)

        for image in os.listdir(self.origin_path):
            self.normalize_single_image(image, size, destin_path, fill_color, delete_original=delete_originals)
        
    def normalize_single_image(self, image, size, destin_path, fill_color=(255, 255, 255), delete_original=False):
        if self.debug:
            print ("Normalizing:", image)

        try:
            origin = self.origin_path + "/" + image
            img = Image.open(origin).convert("RGBA")
            img.thumbnail(size, Image.ANTIALIAS)
            new_img = Image.new('RGB', size, fill_color)
#            new_img = Image.new('RGBA', size, fill_color)
            posx = (int)((size[0] - img.size[0]) / 2)
            posy = (int)((size[1] - img.size[1]) / 2)
            filename, file_extension = os.path.splitext(image)
            new_img.paste(img, (posx, posy), img)
#            new_img.paste(img, (posx, posy))
            new_img.save(destin_path + "/" + filename + ".jpg", "JPEG") #"PNG"
    
            if delete_original:
                os.remove(origin)

        except:
            if self.debug:
                print ("Something goes wrong trying to process image ", image, ":", sys.exc_info()[0])