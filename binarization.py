# 图片二值化
from PIL import Image

image_path = "/home/bit/why/Synthetic_Chinese_License_Plates/pic/"


img = Image.open(image_path+'云A0VHQN.jpg')
 
# 模式L”为灰色图像，它的每个像素用8个bit表示，0表示黑，255表示白，其他数字表示不同的灰度。
Img = img.convert('L')
#Img.save("/home/bit/why/test_云A0VHQN.jpg")
 
# 自定义灰度界限，大于这个值为黑色，小于这个值为白色
threshold = 160
 
table = []
for i in range(256):
    if i < threshold:
        table.append(0)
    else:
        table.append(255)
 
# 图片二值化
photo = Img.point(table, '1')
photo.save("/home/bit/why/test_云A0VHQN.jpg")
