import sys
from io import StringIO
import os
from PIL import Image, ImageFont, ImageDraw
import pygame

pygame.init()

text = u"这是一段测试文本，test 123。"

im = Image.new("RGB", (300, 50), (255, 255, 255))
#dr = ImageDraw.Draw(im)
#font = ImageFont.truetype(os.path.join("fonts", "simsun.ttc"), 18)
font = pygame.font.Font(os.path.join("home/zijian/Probing_Font4Commonsense/src", "simsun.ttc"), 14)

#dr.text((10, 5), text, font=font, fill="#000000")
rtext = font.render(text, True, (0, 0, 0), (255, 255, 255))

#pygame.image.save(rtext, "t.gif")
sio = StringIO.StringIO()
pygame.image.save(rtext, sio)
sio.seek(0)

line = Image.open(sio)
im.paste(line, (10, 5))

im.show()
im.save("t.png")