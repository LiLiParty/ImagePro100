from skimage.util.dtype import img_as_bool
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

st.title("画像処理１００本ノック")
st.write("画像処理について知りたい！研究の前に勉強しておきたい！ということで、画像処理100本ノックをやります")
link_1 = '[画像処理100本ノック!!](https://github.com/yoyoyo-yo/Gasyori100knock)'
st.markdown(link_1, unsafe_allow_html=True)

st.write("※尚、画像処理に非常に時間がかかるものがあります。")

contents = []
img_orig = io.imread("imori_128x128.png") #img_orig[256][256][3]
#img_orig.shape >> (256,256,3)
#img_aft = np.zeros(img_orig.shape)
#img_aft.astype(np.uint8)aa


def RGB_to_BGR(img):
    img_bgr = np.zeros(img.shape)
    a = 2
    for i in range(128):
        for j in range(128):
            img_bgr[i][j][0] = img[i][j][2]
            img_bgr[i][j][1] = img[i][j][1]
            img_bgr[i][j][2] = img[i][j][0]
    return img_bgr.astype(np.uint8)

def Grayscale(img):
    img_aft = np.zeros([128,128])
    #img = img.copy().astype(np.float32)
    for i in range(128):
        for j in range(128):
            img_aft[i][j] += 0.2126*img[i][j][0]
            img_aft[i][j] += 0.7156*img[i][j][1]
            img_aft[i][j] += 0.0722*img[i][j][2]
    img_aft = np.clip(img_aft, 0, 255)
    return img_aft.astype(np.uint8)

def Binarization(img):
    img_aft = Grayscale(img)
    for i in range(128):
        for j in range(128):
            if img_aft[i][j] < 128 :
                img_aft[i][j] = 0
            else:
                img_aft[i][j] = 255
    return img_aft.astype(np.uint8)

def Binarization_otsu(img):
    img_aft = Grayscale(img)
    best_tsd = 0
    best_dis = 0
    for tsd in range(50,207): #時間短縮のため
        left=0
        right=0
        M_left=0
        M_right=0
        for i in range(128):
            for p in img_aft[i]:
                if p <= tsd:
                    left += 1
                    M_left += p
                else:
                    right +=1
                    M_right += p
        M_left /= left
        M_right /= right
        dis = left*right/((left+right)*(left+right))*(M_left-M_right)*(M_left-M_right)
        best_dis = max(dis,best_dis)
        if dis == best_dis : best_tsd = tsd
    for i in range(128):
        for j in range(128):
            if img_aft[i][j] <= best_tsd:
                img_aft[i][j] = 0
            else:
                img_aft[i][j] = 255
    return img_aft.astype(np.uint8)

def Question(num,title,func):
    st.header(title)
    contents.append(title)
    process = st.button("Process:" + num)
    if(process):
        col1, col2= st.beta_columns(2)
        with col1:
            st.write("Before")
            st.image(img_orig,width=150)
        with col2:
            st.write("After")
            img_bgr = func(img_orig)
            st.image(img_bgr,width=150)
    st.write("")


Question("1", "Q.1 チャネル入れ替え", RGB_to_BGR)
"""
```python
def RGB_to_BGR(img):
    img_bgr = np.zeros(img.shape)
    a = 2
    for i in range(128):
        for j in range(128):
            img_bgr[i][j][0] = img[i][j][2]
            img_bgr[i][j][1] = img[i][j][1]
            img_bgr[i][j][2] = img[i][j][0]
    return img_bgr.astype(np.uint8)
```
"""

Question("2", "Q.2 グレースケール", Grayscale)
"""
```python
def Grayscale(img):
    img_aft = np.zeros([128,128])
    #img = img.copy().astype(np.float32)
    for i in range(128):
        for j in range(128):
            img_aft[i][j] += 0.2126*img[i][j][0]
            img_aft[i][j] += 0.7156*img[i][j][1]
            img_aft[i][j] += 0.0722*img[i][j][2]
    img_aft = np.clip(img_aft, 0, 255)
    return img_aft.astype(np.uint8)
```
"""

Question("3", "Q.3 二値化", Binarization)
"""
```python
def Binarization(img):
    img_aft = Grayscale(img)
    for i in range(128):
        for j in range(128):
            if img_aft[i][j] < 128 :
                img_aft[i][j] = 0
            else:
                img_aft[i][j] = 255
    return img_aft.astype(np.uint8)
```
"""

Question("4", "Q.4 大津の二値化", Binarization_otsu)
"""
```python
def Binarization_otsu(img):
    img_aft = Grayscale(img)
    best_tsd = 0
    best_dis = 0
    for tsd in range(50,207): #時間短縮のため
        left=0
        right=0
        M_left=0
        M_right=0
        for i in range(128):
            for p in img_aft[i]:
                if p <= tsd:
                    left += 1
                    M_left += p
                else:
                    right +=1
                    M_right += p
        M_left /= left
        M_right /= right
        dis = left*right/((left+right)*(left+right))*(M_left-M_right)*(M_left-M_right)
        best_dis = max(dis,best_dis)
        if dis == best_dis : best_tsd = tsd
    for i in range(128):
        for j in range(128):
            if img_aft[i][j] <= best_tsd:
                img_aft[i][j] = 0
            else:
                img_aft[i][j] = 255
    return img_aft.astype(np.uint8)
```
"""


st.sidebar.header("もくじ")
for i in range(len(contents)):
    st.sidebar.write(contents[i])