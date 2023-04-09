#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 21:06:13 2022

@author: roux
"""


import cv2 
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('f7.tif') 
imrvb=img.copy()


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     

kernel = np.ones((9,9),np.uint8)
#gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

gray = cv2.GaussianBlur(gray,(9,9),0)

#gray = cv2.medianBlur(gray, ksize=13)

plt.figure()
plt.imshow(gray)
plt.show()
  
edges = cv2.Canny(gray,20,180,apertureSize =3) 
  

  
plt.figure('Canny edges')
plt.imshow(edges)
plt.show()

    
lines = cv2.HoughLines(edges,1,2*np.pi/180,100) 

nbrlines = lines.shape[0]

print('nombre de lignes : ', lines)

print('nombre de lignes : ', nbrlines)


for line1 in range(nbrlines):  
    r1,theta1 = lines[line1][0]
    print(r1,theta1)

nbrintersects=0

fichier = open("data.txt", "w")

for line1 in range(nbrlines):  
    r1,theta1 = lines[line1][0]
    
    a1 = np.cos(theta1) 
    b1 = np.sin(theta1)
    c1 = -r1
    
    x1 = int(a1*r1 + 4000*(-b1)) 
    y1 = int(b1*r1 + 4000*(a1)) 
    x2 = int(a1*r1 - 4000*(-b1)) 
    y2 = int(b1*r1 - 4000*(a1)) 
          
    (a,b,c)=125+125*np.random.rand(3, 1)
    a=int(0)
    b=int(255)
    c=int(0)
    cv2.line(imrvb,(x1,y1), (x2,y2), (a,b,c),7) 
    cv2.line(edges,(x1,y1), (x2,y2), (a,b,c),7) 
    
    x1=int(a1+5)
    y1=int(b1+5)
    x2=int(a1-5)
    y2=int(b1-5)
    
    a=255
    b=0
    c=0
    
    cv2.line(imrvb,(x1,y1), (x2,y2), (a,b,c),5) 
    cv2.line(edges,(x1,y1), (x2,y2), (a,b,c),5) 
#
#    plt.figure()
#    plt.title('edges hough')
#    plt.imshow(edges)
#    plt.show()
#
#    plt.figure()
#    plt.title('image hough')
#    plt.imshow(imrvb)
#    plt.show()


    for line2 in range(line1):
        
        r2,theta2 = lines[line2][0]

        a2 = np.cos(theta2) 
        b2 = np.sin(theta2)
        c2 = -r2
    
        
        X =   b1 * c2 - c1 * b2
        Y =   c1 * a2 - a1 * c2
        Z =   a1 * b2 - b1 * a2
        

        if Z != 0:
            
            X=X/Z
            Y=Y/Z   
            
            if 0 < X < img.shape[1]:
                if 0 < Y < img.shape[0]:
                                
                    nbrintersects += 1
                    
                    print(int(X),int(Y))
                    
                    fichier.write(str ( int ( X ) ) )
                    fichier.write (" ")
                    fichier.write(str ( int ( Y ) ) ) 
                    fichier.write("\n")
                    
                    x1=int(X+5)
                    y1=int(Y+5)
                    x2=int(X-5)
                    y2=int(Y-5)
                
                    a=255
                    b=0
                    c=0
                
                    cv2.line(imrvb,(x1,y1), (x2,y2), (a,b,c),5) 
                    cv2.line(edges,(x1,y1), (x2,y2), (a,b,c),5) 
        
                    x1=int(X-5)
                    y1=int(Y+5)
                    x2=int(X+5)
                    y2=int(Y-5)
        
                    cv2.line(imrvb,(x1,y1), (x2,y2), (a,b,c),5) 
                    cv2.line(edges,(x1,y1), (x2,y2), (a,b,c),5) 

fichier.close()


print('nbr intersects',nbrintersects)

plt.figure()
plt.title('edges')
plt.imshow(edges)
plt.show()

plt.figure()
plt.title('image')
plt.imshow(imrvb)
plt.show()



cv2.imwrite('linesDetected.jpg', img) 




