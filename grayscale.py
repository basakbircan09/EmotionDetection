from PIL import Image
for num in range(1,201):
	img = Image.open("/home/sam02/Desktop/datasets/FEI Face Database/neutral/"+str(num)+"a.jpg").convert('L')
	img.save("/home/sam02/Desktop/datasets/FEI Face Database/neutral_grayscale/feineutral"+str(num)+".jpg")
