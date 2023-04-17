import cv2
import numpy as np

def find_fiducial_marker(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3:
            triangle_center = np.mean(approx, axis=0)[0]
            return int(triangle_center[0]), int(triangle_center[1])



def find_circle_center(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, radius) in circles:
            return x, y

def find_coordinates(image):
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([50, 50, 255])

    mask = cv2.inRange(image, lower_red, upper_red)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_image = image[y:y + h, x:x + w]

            circle_x, circle_y = find_circle_center(cropped_image)
            circle_center = (x + circle_x, y + circle_y)

            fiducial_x, fiducial_y = find_fiducial_marker(image)

            relative_x = circle_center[0] - fiducial_x
            relative_y = circle_center[1] - fiducial_y

            return relative_x, relative_y
def draw_crosshair(image, center, color, size=10, thickness=2):
    x, y = center
    cv2.line(image, (x - size, y), (x + size, y), color, thickness)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness)

def main():
    image = cv2.imread('dewar_fiducial.jpg')
    result = find_coordinates(image)
    if result is not None:
        x, y = result
        print(f'The coordinates of the dewar hole relative to the fiducial marker are: ({x}, {y})')

        # Draw crosshair in the middle of the dark inner circle
        circle_center = (x + find_fiducial_marker(image)[0], y + find_fiducial_marker(image)[1])
        draw_crosshair(image, circle_center, (255, 0, 0))
    else:
        print("Failed to find coordinates.")
    
    # Draw crosshair in the middle of the triangle
    triangle_center = find_fiducial_marker(image)
    if triangle_center is not None:
        draw_crosshair(image, triangle_center, (255, 0, 0))

    cv2.imshow('Input Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()