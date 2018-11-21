'''
Resize an image using PIL 

GHANMI Helmi 
'''
import PIL
from PIL import Image


#resize with proportional hight and width 
basewidth = 300
img = Image.open('images/blueberries.jpg')
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
img.save('temp_resized_image/resized_image.jpg')


# resize for a fixed hight and width 

basewidth = 338
img = Image.open('images/blueberries.jpg')
wpercent = (basewidth / float(img.size[0]))
#hsize = int((float(img.size[1]) * float(wpercent)))
hsize = img.size[0]
img = img.resize((hsize, basewidth), PIL.Image.ANTIALIAS)
img.save('temp_resized_image/resized_image2.jpg')



# resize based on a fixed hight 
'''baseheight = 560
img = Image.open('images/blueberries.jpg')
hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
img.save('resized_image3.jpg')'''

# resize based on a fixed hight 
baseheight = 960
img = Image.open('images/blueberries.jpg')
hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0])))
img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
img.save('temp_resized_image/resized_image3.jpg')

print('The processing is finished')