import sys
import cv2
import numpy as np

def main():
    print("✅ Project entry point works")
    print("✅ Python version:", sys.version)
    print("✅ OpenCV version:", cv2.__version__)
    print("✅ NumPy version:", np.__version__)

    # simple OpenCV test
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.putText(img, "Traffic Light Test", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imshow("Test Window", img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
