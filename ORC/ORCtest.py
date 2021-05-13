#-- coding: utf-8 --

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')


from PIL import Image
import pytesseract


imagepath = 'C:/Users/shinelon/Desktop/test.jpg'
path = "C:/Users/shinelon/Desktop/test1.png"
print((path))
image = Image.open(imagepath)
content_text=pytesseract.image_to_string(image,lang='chi_tra')
print(content_text)
