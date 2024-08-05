import cv2
from matplotlib import pyplot as plt
import numpy as np


def hit_or_miss(img, S, S_bar):
    """
    Thực hiện phép toán Hit-or-Miss trên ảnh

    Args:
      img: Ảnh đầu vào
      S: Phần tử cấu trúc chính
      S_bar: Phần tử cấu trúc phủ định

    Returns:
      Ảnh kết quả sau phép toán Hit-or-Miss
    """

    # Tính toán phép co với S
    erosion_S = cv2.erode(img, S)

    # Tính toán phép co với S_bar trên ảnh đảo ngược
    img_inv = 255 - img
    erosion_S_bar = cv2.erode(img_inv, S_bar)

    # Kết hợp hai kết quả
    hit_miss = cv2.bitwise_and(erosion_S, erosion_S_bar)

    return hit_miss


# # # Tạo ảnh mẫu
# img = np.zeros((10, 10), dtype=np.uint8)
# img[3:7, 3:7] = 1

# Create a simple binary image with a clear pattern
# img = np.zeros((100, 100), dtype=np.uint8)
# img[30:70, 30:70] = 255

# Đọc ảnh
img = cv2.imread('./testimg/meow_400p.jpg', 0)

# Tạo phần tử cấu trúc chính và phủ định
S = np.array([[0, 1, 0],
              [1, 1, 1],
              [0, 1, 0]], dtype=np.uint8)
S_bar = np.array([[1, 0, 1],
                 [0, 0, 0],
                 [1, 0, 1]], dtype=np.uint8)

# Thực hiện phép toán Hit-or-Miss
result = hit_or_miss(img, S, S_bar)

# # Hiển thị kết quả
# cv2.imshow("Original", img)
# cv2.imshow("Hit-or-Miss", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Hiển thị kết quả bằng Matplotlib
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap='gray')
# plt.title('Original Image')

# plt.subplot(1, 2, 2)
# plt.imshow(result, cmap='gray')
# plt.title('Hit-or-Miss Result')

# plt.show()

# Perform hit-or-miss
result = hit_or_miss(img, S, S_bar)

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Original Image
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')

# Erosion with S
erosion_S = cv2.erode(img, S, iterations=1)
axes[0, 1].imshow(erosion_S, cmap='gray')
axes[0, 1].set_title('Erosion with S')

# Erosion with S' on inverted image
img_inv = 255 - img
erosion_S_bar = cv2.erode(img_inv, S_bar)
axes[0, 2].imshow(erosion_S_bar, cmap='gray')
axes[0, 2].set_title('Erosion with S\' on inverted image')

# Intermediate result (for clarity)
intermediate = cv2.bitwise_and(erosion_S, erosion_S_bar)
axes[1, 0].imshow(intermediate, cmap='gray')
axes[1, 0].set_title('Intermediate Result')

# Hit-or-Miss result
axes[1, 1].imshow(result, cmap='gray')
axes[1, 1].set_title('Hit-or-Miss Result')

# Overlay result on original image
overlay = np.where(result == 255, 255, img)
axes[1, 2].imshow(overlay, cmap='gray')
axes[1, 2].set_title('Overlay')

plt.tight_layout()
plt.show()
