#レポート1-3

import cv2
import numpy as np
import math

img = cv2.imread("./golira.jpg")

height = img.shape[0]
width = img.shape[1]

#スケールサイズ
scale_size = 2

for i in range(4):
    #変更後のサイズ
    scale_h = int(height/scale_size)
    scale_w = int(width/scale_size)

    #画像サイズの変更
    img2 = cv2.resize(img,(scale_w,scale_h))

    #画像サイズの変更
    img2 = cv2.resize(img2,(width,height))

    #スケールサイズの表示
    print(f"scale_size: {scale_size}")

    #Calculate PSNR with OpenCV(RGB)
    PSNR_opencv, _ = cv2.quality.QualityPSNR_compute(img, img2)
    print("    PSNR OpenCV (RGB Average):" + str((PSNR_opencv[0] + PSNR_opencv[1] + PSNR_opencv[2]) / 3))

    #Calculate SSIM with OpenCV(RGB)
    SSIM_opencv, _ = cv2.quality.QualitySSIM_compute(img, img2)
    print("    MSSIM OpenCV (RGB Average):" + str((SSIM_opencv[0] + SSIM_opencv[1] + SSIM_opencv[2]) / 3))

    cv2.imwrite("report1-3-" + str(i+1) + ".jpg", img2)
    
    #1/2, 1/8, 1/32, 1/128にするため
    scale_size = scale_size * 4
    
    #考察のため
    # scale_size = scale_size + 1

#ディスプレイ表示
cv2.imshow("output",img)
cv2.imshow("scale_output",img2)

cv2.waitKey(0)
print("finish.")