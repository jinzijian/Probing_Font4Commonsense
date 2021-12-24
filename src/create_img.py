from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np 
import os
import jieba
import matplotlib.pyplot as plt 

def text2img(word):
	color = 0
	img = np.ones([30,60,3],dtype = 'uint8')*255
	img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	draw = ImageDraw.Draw(img)

	size = img.size
	font = min(size[0]//len(word),size[1]-10)
	ttfront = ImageFont.truetype("msyh.ttf", font)
	w,h = ttfront.getsize(word)
	draw.text(((size[0]-w)//2,(size[1]-h)//2),word,fill=(color,color,color), font=ttfront)
	return np.array(img)

def create_samples(texts,is_word,maxlen):
	imgs = []
	seq_len = []
	for text in texts:
		if is_word:
			text = [x for x in text if x!=' ']
		else:
			text = list(jieba.cut(text))
			text = [x for x in text if x!=' ']
		img = []
		for word in text:
			img.append(text2img(word))
		seq_len.append(len(img))
		if len(img)>maxlen:
			img = img[:maxlen]
		else:
			img_ = np.ones([30,60,3],dtype = 'uint8')*255
			img.extend([img_]*(maxlen-len(img)))

		imgs.append(img)
	return imgs, seq_len

def eda():
	with open("D://b站/闲鱼/alex/XNLI-15way/xnli.15way.orig.tsv",encoding = 'utf-8') as f:
		d = f.readlines()[1:]
	d = [x.split("\t") for x in d]
	length = []
	for d_ in d:
		length.extend([len(list(jieba.cut(x))) for x in d_])
	print(max(length))


if __name__=="__main__":

	s = "وقال ، ماما ، لقد عدت للمنزل .	И той каза : Мамо , у дома съм .	und er hat gesagt , Mama ich bin daheim .	Και είπε , Μαμά , έφτασα στο σπίτι .	And he said , Mama , I 'm home .	Y él dijo : Mamá , estoy en casa .	Et il a dit , maman , je suis à la maison .	और उसने कहा , माँ , मैं घर आया हूं ।	И он сказал : Мама , я дома .	Naye akasema , Mama , niko nyumbani .	และเขาพูดว ่ า , ม ่ าม ๊ า ผมอยู ่ บ ้ าน	Ve Anne , evdeyim dedi .	اور اس نے کہا امّی ، میں گھر آگیا ہوں ۔	Và anh ấy nói , Mẹ , con đã về nhà .	他 说 ， 妈妈 ， 我 回来 了 。"
	# s = s.split(" ")
	# color = 0
	# for i in range(len(s)):
	# 	img = np.ones([30,60,3],dtype = 'uint8')*255
	# 	img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	# 	draw = ImageDraw.Draw(img)

	# 	size = img.size
	# 	font = min(size[0]//len(s[i]),size[1]-10)
	# 	ttfront = ImageFont.truetype("msyh.ttf", font)
	# 	w,h = ttfront.getsize(s[i])
	# 	draw.text(((size[0]-w)//2,(size[1]-h)//2),s[i],fill=(color,color,color), font=ttfront)
		# img.save("pics/%d.jpg"%i)
	imgs = create_samples([s,s,s],True,400)
