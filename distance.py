import cv2

# 变量
# 从摄像头到物体（面部）的距离（测量）
KNOWN_DISTANCE = 76.2  # 厘米
# 真实世界中面部的宽度
KNOWN_WIDTH = 14.3  # 厘米
# 颜色
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX
cap = cv2.VideoCapture(1)

# 面部检测对象
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# 焦距计算函数
def focal_length(measured_distance, real_width, width_in_rf_image):
    """
    这个函数计算焦距（从镜头到CMOS传感器的距离），可以使用测量距离、实际宽度和图像中物体的宽度找到这个常数
    :param1 Measure_Distance(int): 在捕获参考图像时，从物体到摄像头的测量距离

    :param2 Real_Width(int): 物体的实际宽度，在现实世界中（例如我的面部宽度是14.3厘米）
    :param3 Width_In_Image(int): 图像中物体的宽度（通过面部检测器找到的参考图像中的宽度）
    :return focal_length(Float): 返回焦距
    """
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value


# 距离估算函数
def distance_finder(focal_length, real_face_width, face_width_in_frame):
    """
    这个函数简单估算物体和摄像头之间的距离，使用参数（焦距、实际物体宽度、图像中的物体宽度）
    :param1 focal_length(float): 由焦距计算函数返回的值

    :param2 Real_Width(int): 物体的实际宽度，在现实世界中（例如我的面部宽度是5.7英寸）
    :param3 object_Width_Frame(int): 图像中物体的宽度（使用视频流）
    :return Distance(float) : 返回估算的距离
    """
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance


# 面部检测函数
def face_data(image):
    """
    这个函数检测面部
    :param 接受图像作为参数
    :returns 面部宽度（像素）
    """

    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for x, y, h, w in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), WHITE, 1)
        face_width = w

    return face_width


# 从目录中读取参考图像
ref_image = cv2.imread("Ref_image.png")
if ref_image is None:
    print("Error: Could not read the reference image.")
    exit()

ref_image_face_width = face_data(ref_image)
if ref_image_face_width == 0:
    print("Error: Could not detect a face in the reference image.")
    exit()

focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width)
print(f"Focal Length: {focal_length_found}")
cv2.imshow("Reference Image", ref_image)

# 从目录中读取要检测的图像
test_image = cv2.imread("test.jpeg")
if test_image is None:
    print("Error: Could not read the test image.")
    exit()

face_width_in_test_image = face_data(test_image)
if face_width_in_test_image != 0:
    Distance = distance_finder(
        focal_length_found, KNOWN_WIDTH, face_width_in_test_image
    )
    cv2.putText(
        test_image, f"Distance = {round(Distance, 2)} CM", (50, 50), fonts, 1, WHITE, 2
    )
else:
    print("Error: Could not detect a face in the test image.")

cv2.imshow("Test Image", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
