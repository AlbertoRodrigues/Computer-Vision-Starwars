import cv2
import numpy as np
import matplotlib.pyplot as pt
from PIL import Image
import imutils

#Tarefa 1, opencv, pillow e GGPLOT
#Exemplos de imagens
end="Yoda\download36.jpg"
end2="Stormtrooper\download10.jpg"
end3="Darth Vader\download16.jpg"
img=cv2.imread(end)
img2=cv2.imread(end2)
img3=cv2.imread(end3)





def visualiza_imagem(ind,end):
    if ind=="opencv":
        img=cv2.imread(end)
        cv2.imshow("Imagem",img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    elif ind=="pillow" or ind=="matplotlib":
        im = Image.open(end) 
        if ind==2:
            im.show()
        else:
            pt.imshow(im)    
    else:
        print("Opção nao disponivel")
        
visualiza_imagem("pillow" ,end )
   
Image.fromarray(np.floor(0.114*img[:,:,0]+0.587*img[:,:,1]+0.299*img[:,:,2]).astype("uint8")) 



#Tarefa 2


#0.299*r + 0.587*g + 0.114*b.
np.floor(0.114*img[:,:,0]+0.587*img[:,:,1]+0.299*img[:,:,2]).astype("uint8")
def transforma_em_cinza(img):    
    cv2.imshow("Imagem Original",img)
    cv2.imshow("Imagem cinza numpy",np.floor(0.114*img[:,:,0]+0.587*img[:,:,1]+0.299*img[:,:,2]).astype("uint8"))
    cv2.waitKey()
    cv2.destroyAllWindows()
transforma_em_cinza(img)
transforma_em_cinza(img2)
transforma_em_cinza(img3)

cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_original=cv2.cvtColor(cinza, cv2.COLOR_GRAY2RGB)
cv2.imshow("Imagem cinza",cinza)
cv2.imshow("Imagem colorida",img_original)
cv2.waitKey()
cv2.destroyAllWindows()
#Duvida na parte 
#Depois veja como
#converter espaços de cores usando OpenCV
#Sera que eh a imagem na cor de cada canal?
#Sequencia de cores usadas no opencv eh BGR
#Tarefa 3

#inversao
def inversao_imagem(img):    
    cv2.imshow("Imagem Original",img)
    cv2.imshow("Imagem invertida",img[:,::-1,:])
    cv2.waitKey()
    cv2.destroyAllWindows()

inversao_imagem(img)
inversao_imagem(img2)
inversao_imagem(img3)
#Zoom aleatorio
def zoom_aleatorio(img):   
    tamanho_janela=np.int(np.ceil(img.shape[0]/2))
    linha_zoom=np.random.randint(0, img.shape[0]-tamanho_janela)
    coluna_zoom=np.random.randint(0, img.shape[1]-tamanho_janela)
    img_zoom=img[linha_zoom:(linha_zoom+tamanho_janela),coluna_zoom:(coluna_zoom+tamanho_janela),:]
    cv2.imshow("Imagem Original",img)
    cv2.imshow("Zoom aleatorio",cv2.resize(img_zoom,(img.shape[0], img.shape[1])))
    #cv2.imshow("Zoom aleatorio",img_zoom)
    cv2.waitKey()
    cv2.destroyAllWindows()
zoom_aleatorio(img)
zoom_aleatorio(img2)
zoom_aleatorio(img3)
#Ver depois como fazer
#Rotação de um angulo aleatorio
angulo=10
x=84
y=150
a=np.array([[1,2],[3,4]])
int(np.ceil(a*np.cos(angulo)-a*np.sin(angulo)))
int(np.ceil(x*np.cos(angulo)-y*np.sin(angulo)))
int(np.ceil(x*np.sin(angulo)+y*np.cos(angulo)))
#Inversao da imagem
cv2.imshow("Imagem Original",img)
cv2.imshow("Zoom aleatorio",cv2.flip(img,1))
cv2.waitKey()
cv2.destroyAllWindows()
#Rotação
cv2.imshow("Imagem Original",img)
rotated = imutils.rotate_bound(img, 92)
cv2.imshow("Rotated Without Cropping", rotated)
cv2.waitKey()
cv2.destroyAllWindows()
#Brilho e contraste aleatorio
alpha=1
beta=100
new_image = np.zeros(img.shape, img.dtype)
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        for c in range(img.shape[2]):
            new_image[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
cv2.imshow('Original Image', img)
cv2.imshow('New Image', new_image)
# Wait until user press some key
cv2.waitKey()
cv2.destroyAllWindows()
#Zoom aleatorio
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))

#Tarefa 4
def filtro_media(img):
    cv2.imshow("Imagem Original",img)
    cv2.imshow("Filtro media",cv2.blur(img, (5,5) ))
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def filtro_gaussian_blur(img):
    cv2.imshow("Imagem Original",img)
    cv2.imshow("Filtro media gaussiana",cv2.GaussianBlur(img, (5,5) ))
    cv2.waitKey()
    cv2.destroyAllWindows() 
    
def filtro_mediana(img):
    cv2.imshow("Imagem Original",img)
    cv2.imshow("Filtro mediana",cv2.medianBlur(img, (5,5) ))
    cv2.waitKey()
    cv2.destroyAllWindows() 
borrada = cv2.blur(cinza, (5,5) )
gb = cv2.GaussianBlur( cinza, (5,5), 0 )
imgborradamediana=cv2.medianBlur(cinza, 5)