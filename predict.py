# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#
from yolo import YOLO
from PIL import Image

yolo = YOLO()

while True:
    # img = input('Input image filename:')
    try:
        # image = Image.open(img)
        image = Image.open("img/street.jpg")
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
        r_image.save("img/street_p.jpg")

    break