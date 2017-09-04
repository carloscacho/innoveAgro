import cv2
import numpy as np
from pylab import *

contabil = []
#definicoo do intervalo de cores
cores = []
# verde
cores.append([33,187,200])
cores.append([41,176,184])
#amarelo
cores.append([81,221,255])
cores.append([55,174,189])
#marron-escuro
cores.append([47,87,129])
cores.append([81,181,216])
#passar pela imagem uma vez para cada cor
for cor in range(0,3):
    print "cor " + str(cor)

    #carrega uma cor para ser analisada
    blue = cores[cor*2][0]
    green = cores[cor*2][1]
    red = cores[cor*2][2]

    blue2 = cores[cor*2 +1][0]
    green2 = cores[cor*2+1][1]
    red2 = cores[cor*2+1][2]

    #transforma a cor que sera analisada no momento de  RGB em HSF
    color = np.float32([[[blue, green, red]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    color2 = np.float32([[[blue2, green2, red2]]])
    hsv_color2 = cv2.cvtColor(color2, cv2.COLOR_BGR2HSV)

    hue = hsv_color[0][0][0]
    hue2 = hsv_color2[0][0][0]

    #capitura de vido
    #cap = cv2.VideoCapture(0)

    # while(1):

    # Coleta dos frames
    # _, frame = cap.read()
    frame  = cv2.imread('amosta.jpeg')

    #jogar fora metade das imagens no video
    # jogafora = True

    # converter a imagen coletada em  de rgb em hsv
    hsv = (frame/255.0).astype(np.float32)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    # define o total de cores que sera analisada no momento HSV
    lower = np.array([hue-2,0,0])
    upper = np.array([hue2+2,1,1])

    # aplicacao de smothing
    kernel = np.ones((5,5),np.float32)/25
    hsv = cv2.filter2D(hsv,-1,kernel)
    # aplicacao de blur na imagem
    hsv = cv2.blur(hsv,(5,5))

    # aplicacao de um trashold na imagem capturando para remocao do fundo
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    img_rgb = res

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    if cor == 0:
        template = cv2.imread('grao.png',0)
        threshold = 0.6
    if cor == 1:
        template = cv2.imread('graoBomTest.png',0)
        threshold = 0.75
    if cor == 2:
        template = cv2.imread('graoMarronTest.png',0)
        threshold = 0.6
    w, h = template.shape[::-1]

    rescir = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    #threshold = 0.75
    loc = np.where( rescir >= threshold)
    count = 0
    listPoit = []
    #busca dos elementos similares
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
    # exibicao da imagen original
    cv2.imshow('original',frame)

    # exibicao da imagen pela separacao de cor
    if cor == 0:
        cv2.imshow('Verde',res)
    if cor == 1:
        cv2.imshow('Bom',res)
    if cor == 2:
        cv2.imshow('Marron',res)
    print count
    contabil.append(count)

#calculo dos outros graos que ainda nao foram verificados
contar = 0
for ct in contabil:
    contar += ct
contabil.append(20 - contar)

#geracao do grafico em pizza
figure(1, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])
labels = 'verde', 'bom', 'marron','outros'
fracs = contabil
explode=(0, 0.05, 0, 0)
pie(fracs, explode=explode, labels=labels,autopct='%1.1f%%', shadow=True, startangle=90)
title('Analise dos graos', bbox={'facecolor':'0.8', 'pad':5})
show()
#comando para finalizar o codigo
while True:
    ch = cv2.waitKey(27)
    if ch == 27:#Escape
        cv2.destroyAllWindows()
        break
