import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
start_time = time.time()


### Set the variables
IMAGE_PATH = "polaris.png" #7.27
#IMAGE_PATH = "website.png" #11.50

def Image_To_Text():
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(IMAGE_PATH)

def Pre_Process_Image():
    print("Processing image")
    original = cv2.imread(IMAGE_PATH)
    cv2.imwrite('original.png', original)

    imagem = cv2.bitwise_not(original)
    cv2.imwrite('bitwise_not.png', imagem)

    imagem2 = cv2.bitwise_not(imagem)
    cv2.imwrite('bitwise.png', imagem2)

    blur1 = cv2.blur(original,(5,5))
    cv2.imwrite('blur1.png', blur1)

    GaussianBlur = cv2.GaussianBlur(original, (5, 5), 0)
    cv2.imwrite('GaussianBlur.png', GaussianBlur)

    medianBlur = cv2.medianBlur(original, 3)
    cv2.imwrite('medianBlur.png', medianBlur)

    bilateralFilter = cv2.bilateralFilter(original,9,75,75)
    cv2.imwrite('bilateralFilter.png', bilateralFilter)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(original,kernel,iterations = 1)
    cv2.imwrite('erosion.png', erosion)

    dilation = cv2.dilate(original,kernel,iterations = 1)
    cv2.imwrite('dilation.png', dilation)

    opening = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('opening.png', dilation)

    closing = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('closing.png', dilation)

    gradient = cv2.morphologyEx(original, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite('gradient.png', dilation)

    tophat = cv2.morphologyEx(original, cv2.MORPH_TOPHAT, kernel)
    cv2.imwrite('tophat.png', dilation)

    blackhat = cv2.morphologyEx(original, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite('blackhat.png', dilation)


Pre_Process_Image()

def Find_Contours():
    names = ["original.png", "blur1.png", "GaussianBlur.png", "medianBlur.png", "bilateralFilter.png", "erosion.png", "dilation.png"]

    for name in names:
        try:
            large = cv2.imread(name)
            rgb = None

            try:
                rgb = cv2.pyrDown(large)
                pass
            except Exception as e:
                break

            small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
            _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mask = np.zeros(bw.shape, dtype=np.uint8)
            for idx in range(len(contours)):
                x, y, w, h = cv2.boundingRect(contours[idx])
                mask[y:y+h, x:x+w] = 0
                cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
                r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
                if r > 0.45 and w > 5 and h > 5:
                    cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

            cv2.imwrite(name + "_contours.png", rgb)
            pass
        except Exception as e:
            print("exception happened " + name)
            print(e)
            raise

def edge_detect():
    names = ["bitwise_not.png", "original.png", "blur1.png", "GaussianBlur.png", "medianBlur.png", "bilateralFilter.png", "erosion.png", "dilation.png"]
    for name in names:
        image = cv2.imread(name)
        im_bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        (thresh, im_bw) = cv2.threshold(im_bw, 128, 255, 0)
        cv2.imwrite('bw_'+name, im_bw)

        contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0,255,0), 3)
        cv2.imwrite('edge_'+name, image)


def Find_Contours2():
    names = ["original.png", "blur1.png", "GaussianBlur.png", "medianBlur.png", "bilateralFilter.png", "erosion.png", "dilation.png"]

    for name in names:
        try:
            large = cv2.imread(name)
            rgb = None

            try:
                rgb = cv2.pyrDown(large)
                pass
            except Exception as e:
                break

            small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
            _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            mask = np.zeros(bw.shape, dtype=np.uint8)
            for idx in range(len(contours)):
                x, y, w, h = cv2.boundingRect(contours[idx])
                mask[y:y+h, x:x+w] = 0
                cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
                r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
                if r > 0.001 and w > 0.01 and h > 0.01:
                    cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)

            cv2.imwrite("try2_" + name + "_contours.png", rgb)
            pass
        except Exception as e:
            print("exception happened " + name)
            print(e)
            raise

if __name__ == "__main__":
    Find_Contours()
    Find_Contours2()
    edge_detect()
