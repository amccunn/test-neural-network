from PIL import Image

img = Image.open("animal-icon.png")
img.save("animal-icon.ico", format="ICO", sizes=[(256, 256)])