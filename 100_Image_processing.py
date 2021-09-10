from skimage.util.dtype import img_as_bool
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

st.title("画像処理いろいろ")
st.write("画像処理について知りたい！研究の前に勉強しておきたい！ということで、「画像処理100本ノック!!」をやってみるページ")
st.write("引用・参考サイトは以下です。")
link_1 = '[画像処理100本ノック!!](https://github.com/yoyoyo-yo/Gasyori100knock)'
st.markdown(link_1, unsafe_allow_html=True)

st.write("※画像処理に非常に時間がかかるものがあります。")

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

def HSV_translate(img):
    img_aft = img
    for i in range(128):
        for j in range(128):
            R,G,B = img_aft[i][j][0]/255,img_aft[i][j][1]/255,img_aft[i][j][2]/255
            Vmax,max_RGB  = max([ R,G,B ]),np.argmax([ R,G,B ])
            Vmin,min_RGB = min([ R,G,B ]),np.argmin([ R,G,B ])
            if Vmax == Vmin: Hd = 0
            elif min_RGB == 0: Hd = (60*(B-G)/(Vmax-Vmin) + 180) / 60 
            elif min_RGB == 1: Hd = (60*(R-B)/(Vmax-Vmin) + 300) / 60
            elif min_RGB == 2: Hd = (60*(G-R)/(Vmax-Vmin) + 60)  / 60
            Hd = int(Hd)
            if(Hd > 5): Hd = 5
            S,V = Vmax - Vmin,Vmax
            C = S
            X = C*(1-abs(Hd%2 - 1))
            #if(i==1): st.write(X)
            #Hd,V,S,X
            toRGB = [[C,X,0],[X,C,0],[0,C,X],[0,X,C],[X,0,C],[C,0,X]];
            #if(i==1): st.write(Vmin + toRGB[Hd][0])
            img_aft[i][j][0],img_aft[i][j][1],img_aft[i][j][2] = V-C + toRGB[Hd][0], V-C + toRGB[Hd][1], V-C + toRGB[Hd][2]            
            img_aft[i][j][0] *= 255
            img_aft[i][j][1] *= 255
            img_aft[i][j][2] *= 255
    return img_aft

def hsv2rgb(hsv):
    h, s, v = np.split(hsv, 3, axis=2)
    h, s, v = h[..., 0], s[..., 0], v[..., 0]
    _h = h / 60
    x = s * (1 - np.abs(_h % 2 - 1))
    z = np.zeros_like(x)
    vals = np.array([[s, x, z], [x, s, z], [z, s, x], [z, x, s], [x, z, s], [s, z, x]])
    
    img = np.zeros_like(hsv)
    
    for i in range(6):
        ind = _h.astype(int) == i
        for j in range(3):
            img[..., j][ind] = (v - s)[ind] + vals[i, j][ind]
            
    return np.clip(img, 0, 255).astype(np.uint8)

def rgb2hsv(img):
    _img = img.copy().astype(np.float32)# / 255
    v_max = _img.max(axis=2)
    v_min = _img.min(axis=2)
    v_argmin = _img.argmin(axis=2)
    hsv = np.zeros_like(_img, dtype=np.float32)
    r, g, b = np.split(_img, 3, axis=2)
    r, g, b = r[..., 0], g[..., 0], b[..., 0]

    diff = np.maximum(v_max - v_min, 1e-10)
    
    # Hue
    ind = v_argmin == 2
    hsv[..., 0][ind] = 60 * (g - r)[ind] / diff[ind] + 60
    ind = v_argmin == 0
    hsv[..., 0][ind] = 60 * (b - g)[ind] / diff[ind] + 180
    ind = v_argmin == 1
    hsv[..., 0][ind] = 60 * (r - b)[ind] / diff[ind] + 300
    ind = v_max == v_min
    hsv[..., 0][ind] = 0
    # Saturation
    hsv[..., 1] = v_max - v_min
    # Value
    hsv[..., 2] = v_max
    hsv[..., 0] = (hsv[..., 0] + change) % 360
    return hsv2rgb(hsv)

def Question(num,title,func):
    st.header(title)
    contents.append(title)
    process = st.button("Process:" + num)
    if(process):
        col1, col2= st.columns(2)
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
change = st.slider('色相変化', min_value=0, max_value=180)
Question("5", "Q.5 HSV変換で色相(Hue)反転 要修正", rgb2hsv)
"""
```python
#以下のコードHSV変換はうまく動いていません。引用元サイトのコードで画像処理は動かしています。
def HSV_translate(img):
    img_aft = img
    for i in range(128):
        for j in range(128):
            R,G,B = img_aft[i][j][0]/255,img_aft[i][j][1]/255,img_aft[i][j][2]/255
            Vmax,max_RGB  = max([ R,G,B ]),np.argmax([ R,G,B ])
            Vmin,min_RGB = min([ R,G,B ]),np.argmin([ R,G,B ])
            if Vmax == Vmin: Hd = 0
            elif min_RGB == 0: Hd = (60*(B-G)/(Vmax-Vmin) + 180) / 60 
            elif min_RGB == 1: Hd = (60*(R-B)/(Vmax-Vmin) + 300) / 60
            elif min_RGB == 2: Hd = (60*(G-R)/(Vmax-Vmin) + 60)  / 60
            Hd = int(Hd)
            if(Hd > 5): Hd = 5
            S,V = Vmax - Vmin,Vmax
            C = S
            X = C*(1-abs(Hd%2 - 1))
            #if(i==1): st.write(X)
            #Hd,V,S,X
            toRGB = [[C,X,0],[X,C,0],[0,C,X],[0,X,C],[X,0,C],[C,0,X]];
            #if(i==1): st.write(Vmin + toRGB[Hd][0])
            img_aft[i][j][0],img_aft[i][j][1],img_aft[i][j][2] = V-C + toRGB[Hd][0], V-C + toRGB[Hd][1], V-C + toRGB[Hd][2]            
            img_aft[i][j][0] *= 255
            img_aft[i][j][1] *= 255
            img_aft[i][j][2] *= 255
    return img_aft
```
"""

st.sidebar.header("もくじ")
for i in range(len(contents)):
    st.sidebar.write(contents[i])