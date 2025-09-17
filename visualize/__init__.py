import cv2
import numpy as np

def visualization(image, label, base):
    print(base)
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_np = np.ascontiguousarray(img_np)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for x, y in label.cpu().numpy():
        cv2.circle(img_np, (int(x), int(y)), radius=3, color=(255, 0, 255), thickness=-1)

    cv2.imshow("Tile", img_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()