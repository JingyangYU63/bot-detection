import cv2
import numpy as np
import unittest

def post_process(

    img: np.ndarray[np.uint8],

    msk: np.ndarray[np.uint8]

) -> tuple[tuple[float, float], float, np.ndarray[np.uint8]]:
    
    # Tranform the image into black and white format
    _, msk = cv2.threshold(msk, 120, 255, cv2.THRESH_BINARY)
    
    # Find the contours
    contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the moments
    moments = cv2.moments(contours[0])

    # Calculate the centroid
    centroid_x = int(moments['m10'] / moments['m00'])
    centroid_y = int(moments['m01'] / moments['m00'])

    # Calculate the angle of orientation
    (x, y), (MA, ma), angle = cv2.fitEllipse(contours[0])

    print('Centroid:', (centroid_x, centroid_y))
    print('Angle of orientation:', angle)
    
    # Draw the centroid on the image as a green point
    cv2.circle(img, (centroid_x, centroid_y), 1, (0, 255, 0), -1)

    # Draw the orientation angle in blue
    angle_rad = angle * (np.pi/180)  # Convert angle to radians
    x1 = int(x - 100 * np.cos(angle_rad))
    y1 = int(y + 100 * np.sin(angle_rad))
    x2 = int(x + 100 * np.cos(angle_rad))
    y2 = int(y - 100 * np.sin(angle_rad))
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    # Display the image
    cv2.imshow("Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return ((centroid_x, centroid_y), angle, img)

def get_centroid_and_angle(img):
    
    cv2.imshow("Original image", img)
    cv2.waitKey(0)
    
    B, G, R = cv2.split(img)
    
    # Threshold the image
    _, thresh_B = cv2.threshold(B, 150, 255, cv2.THRESH_BINARY)
    _, thresh_G = cv2.threshold(G, 150, 255, cv2.THRESH_BINARY)
    
    # Display the image
    cv2.imshow("Blue", thresh_B)
    cv2.waitKey(0)
    cv2.imshow("Green", thresh_G)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find the contours in the binary image
    contours_B, _ = cv2.findContours(thresh_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_G, _ = cv2.findContours(thresh_G, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the moments of the contours
    moments = cv2.moments(contours_G[0])

    # Calculate the centroid of the green point
    centroid_x = int(moments['m10'] / moments['m00'])
    centroid_y = int(moments['m01'] / moments['m00'])

    # Calculate the angle of the blue line
    rows,cols = img.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(contours_B[0], cv2.DIST_L2,0,0.01,0.01)
    angle = 180 - np.rad2deg(np.arctan2(vy, vx))[0]
    
    return ((centroid_x, centroid_y), angle, img)

class TestPostProcess(unittest.TestCase):
    
    def setUp(self):
        # Perform any initialization required before each test method
        
        # To read image from disk, we use
        # cv2.imread function, in below method
        self.img = cv2.imread("input_0.png")
        self.msk = cv2.imread("input_1.png", cv2.IMREAD_GRAYSCALE)
        self.out = cv2.imread("output.png")
        
    def tearDown(self):
        # Perform any cleanup required after each test method
        del self.img
        del self.msk
        del self.out
    
    def test_post_process(self):
    
        centroid, angle, img = post_process(self.img, self.msk)
        centroid_x, centroid_y = centroid
        
        centroid_exp, angle_exp, img_exp = get_centroid_and_angle(self.out)
        centroid_x_exp, centroid_y_exp = centroid_exp
        
        self.assertAlmostEqual(centroid_x, centroid_x_exp)
        self.assertAlmostEqual(centroid_y, centroid_y_exp)
        self.assertAlmostEqual(angle, angle_exp, delta=15) # assert equality in tolorence of 15 degrees

if __name__ == "__main__":
    
    unittest.main()