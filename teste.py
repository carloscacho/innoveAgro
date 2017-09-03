import cv2
import numpy as np
from pylab import *

hsf_select = 0;
min_value = 100;
max_value = 255;
select = True;
troquei = False;


# verde
cores = []
cores.append([33,187,200])
cores.append([41,176,184])
#amarelo
cores.append([81,221,255])
cores.append([55,174,189])
#marron-escuro
cores.append([47,87,129])
cores.append([81,181,216])

contabil = []
for cor in range(0,3):
    print "cor " + str(cor)


    # blue = float(sys.argv[1])/255.0
    # green = float(sys.argv[2])/255.0
    # red = float(sys.argv[3])/255.0
    #
    # blue2 = float(sys.argv[4])/255.0
    # green2 = float(sys.argv[5])/255.0
    # red2 = float(sys.argv[6])/255.0


    blue = cores[cor*2][0]
    green = cores[cor*2][1]
    red = cores[cor*2][2]

    blue2 = cores[cor*2 +1][0]
    green2 = cores[cor*2+1][1]
    red2 = cores[cor*2+1][2]

    color = np.float32([[[blue, green, red]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    color2 = np.float32([[[blue2, green2, red2]]])
    hsv_color2 = cv2.cvtColor(color2, cv2.COLOR_BGR2HSV)

    hue = hsv_color[0][0][0]
    hue2 = hsv_color2[0][0][0]

    #cap = cv2.VideoCapture(0)

    # while(1):

    # Take each frame

    # _, frame = cap.read()
    frame  = cv2.imread('soja6.jpeg')

    # jogafora = True

    #ponto flutuante
    #h ,w, b = frame.shape

    hsv = (frame/255.0).astype(np.float32)
    #print hsv[0][0][0]
    # Convert BGR to HSV
    #hsv = np.zeros(frame.shape)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    #print hsv[0][0][0]

    # define range of blue color in HSV
    lower_blue = np.array([hue-2,0,0])
    upper_blue = np.array([hue2+2,1,1])

    #smothing
    kernel = np.ones((5,5),np.float32)/25
    hsv = cv2.filter2D(hsv,-1,kernel)
    hsv = cv2.blur(hsv,(5,5))


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    img_rgb = res
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    if cor == 0:
        template = cv2.imread('grao.png',0)
    if cor == 1:
        template = cv2.imread('graoBom.png',0)
    if cor == 2:
        template = cv2.imread('graoBom.png',0)
    w, h = template.shape[::-1]

    rescir = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    loc = np.where( rescir >= threshold)
    count = 0
    listPoit = []
    for pt in zip(*loc[::-1]):
        if listPoit == []:
            listPoit.append(pt);
            count = 1
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        else:
            contem = False
            for i in listPoit:
                x, y = i
                ptx, pty = pt
                if ((x - w-2 < ptx) and (ptx < x+ w+2)) and ((y- h-2 < pty) and (pty < y+ h+2)):
                    #print " ja existe"
                    contem = True

            if not contem :
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                listPoit.append(pt)
                count += 1
    #cv2.imshow('circulos',img_rgb)
    if cor == 0:
        cv2.imshow('Verde',res)
    if cor == 1:
        cv2.imshow('Bom',res)
    if cor == 2:
        cv2.imshow('Marron',res)
    print count
    contabil.append(count)
    # cv2.imshow('frame',frame)
    # cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
print contabil
# make a square figure and axes
figure(1, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])

# The slices will be ordered and plotted counter-clockwise.
labels = 'verde', 'bom', 'marron'
fracs = contabil
explode=(0, 0.05, 0)

pie(fracs, explode=explode, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.

title('Analise dos graos', bbox={'facecolor':'0.8', 'pad':5})

show()
while True:
    ch = cv2.waitKey(27)
    if ch == 27:#Escape
        cv2.destroyAllWindows()

        break
