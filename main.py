import tensorflow as tf
from tensorflow.keras import models, layers

import cv2
import numpy as np

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

def process_image(img):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()
    fig.tight_layout()
    fig.canvas.manager.set_window_title('IMAGE PROCESSING')
    
    k = 0

    axs[k].imshow(img, cmap='gray')
    axs[k].set_title('original')
    axs[k].axis('off')
    k += 1
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    axs[k].imshow(img, cmap='gray')
    axs[k].set_title('to gray')
    axs[k].axis('off')
    k += 1

    img = cv2.convertScaleAbs(img, alpha=1, beta=210)
    
    axs[k].imshow(img, cmap='gray')
    axs[k].set_title('brightness')
    axs[k].axis('off')
    k += 1
    
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    axs[k].imshow(img, cmap='gray')
    axs[k].set_title('contrast')
    axs[k].axis('off')
    k += 1

    img = cv2.bitwise_not(img)
    axs[k].imshow(img, cmap='gray')
    axs[k].set_title('invert')
    axs[k].axis('off')
    k += 1
    
    ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY) 
    axs[k].imshow(img, cmap='gray')
    axs[k].set_title('binary thresh')
    axs[k].axis('off')
    k += 1

    return img


def main(filename):
    model = models.load_model('ckpt\\cp-06.keras')
    
    img = cv2.imread(filename)
    pimg = process_image(img)

    contours, hirarchy = cv2.findContours(pimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fig, axs = plt.subplots(1, 3, figsize=(15, 8))
    axs = axs.flatten()
    fig.tight_layout()

    axs[0].imshow(img)
    axs[0].set_title('original')
    axs[0].axis('off')

    axs[1].imshow(pimg, cmap='gray')
    axs[1].set_title('proccessed')
    axs[1].axis('off')
    
    axs[2].imshow(pimg, cmap='gray')
    axs[2].set_title('contours')
    axs[2].axis('off')

    detected = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 20 and w * h < 65000 and min(w, h) / max(w, h) > 0.2:
            if x - 10 >= 0:
                x = x - 10
            else:
                x = 0

            if y - 10 >= 0:
                y = y - 10
            else:
                y = 0

            w = w + 10
            h = h + 10
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', fill=False)
            axs[2].add_patch(rect)

            roi = pimg[y:y+h, x:x+w]
            roi = cv2.resize(roi, (12, 18))
            roi = cv2.copyMakeBorder(roi, 5, 5, 8, 8, cv2.BORDER_CONSTANT, value=0)
            ret, thresh = cv2.threshold(roi, 1, 255, cv2.THRESH_BINARY)
            detected.append((thresh, (x, y, w, h)))

    n, m = 1, len(detected)
    for i in range((len(detected) - 1) // 2, -1, -1):
        if (len(detected) % i) == 0:
            n = i
            m = len(detected) // i
            break
    
    dfig, axsd = plt.subplots(n, m, figsize=(8, 8), squeeze=False)
    axsd = axsd.flatten()
    dfig.tight_layout()
    dfig.canvas.manager.set_window_title('DETECTED')

    detected = sorted(detected, key=lambda d: (d[1][1] // 50, d[1][0]))
    
    matrix = []

    k = 0
    for obj, pos in detected:
        axsd[k].imshow(obj, cmap='gray')
        obj = obj.astype('float64') / 255.0
        obj = obj.reshape(1, 28, 28, 1)
        preds = model.predict(obj)
        
        final = np.argmax(preds)
        matrix.append(final)

        axs[2].annotate(f'{final}', (pos[0], pos[1]), color='lightblue')
        print(f'Prediction: {final} at ({pos[0]},  {pos[1]}) size ({pos[2]}, {pos[3]})')

        axsd[k].set_title(f'{final}')
        k += 1

    n = math.sqrt(len(detected))
    if not n.is_integer():
        print('[!] ERROR: image problem')
        print('\t - possible not well lit')
        print('\t - possible not flat')
        print('\t - possible writting is bad')
        print('\t - possible not n x n matrix')
        return
    
    n = int(n)

    matrix = np.reshape(matrix, (n, n))
    print(f'{n}x{n}')
    print(f'A={matrix}') 

    determinant = np.linalg.det(matrix)
    determinant = round(determinant, 2)
    print(f'\n\nDETERMINANT: {determinant}')

    plt.show()
    
if __name__ == '__main__':
    print('[!] NOTICE')
    print('\t- file must be in the same folder')
    print('\t- use plain white paper and black pen')
    
    name = 'media\\' + input('filename: ')

    if os.path.isfile(name): 
        print(f'file {name} exists')
        print('continueing...')
        main(name)
    else: 
        print(f'file {name} missing')
    
    print('exitted...')

