import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

# workoncv-4.1.0
# ipython

class GreenScreenKeying:
    def __init__(self, windowName, image, background):
        self.windowName = windowName
        self.image = image
        self.HSVImage = image
        self.workingImage = image
        self.background = background
        self.kernalSize = 5
        self.greenSpillScale = 1
        self.upperThreshold = None
        self.lowerThreshold = None
        self.hsv = ()
        cv2.imshow(self.windowName, self.image)
        cv2.setMouseCallback(self.windowName, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.edit_image(x, y)

    def edit_image(self, x, y):
        # grab the color clicked
        self.hsv = self.pixel_selector(x, y)
        # remove the mouse call back
        cv2.setMouseCallback(self.windowName, lambda *args: None)
        # create default color 
        self.upper, self.lower = self.color_selector(20)
        # update image and display
        self.update_image()
        # set trackbar for color selector/mask blurring/green spill removal
        self.add_trackbars()

    def pixel_selector(self, x, y):
        # convert image to hsv
        self.HSVImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # grab h,s,v from selected pixel
        h, s, v = self.HSVImage[y, x]
        return (h, s, v)

    def color_selector(self, tolerance):
        h, s, v = self.hsv
        # grab a range of hsv around this color
        hue_max = self.adj_tolerance(h, 15, 180, True)
        hue_min = self.adj_tolerance(h, 15, 180, False)
        sat_max = self.adj_tolerance(s, tolerance, 255, True)
        sat_min = self.adj_tolerance(s, tolerance, 255, False)
        val_max = self.adj_tolerance(v, tolerance, 255, True)
        val_min = self.adj_tolerance(v, tolerance, 255, False)
        # create a mask based off this range
        upper = np.array([hue_max, sat_max, val_max])
        lower = np.array([hue_min, sat_min, val_min])
        self.upper = upper
        self.lower = lower
        return (upper, lower)

    def adj_tolerance(self, value, tolerance, maxVal, upper):
        return (value + tolerance) % maxVal if upper else (value - tolerance) % maxVal

    def remove_background(self, image_mask):
        # blur the mask
        image_mask = cv2.GaussianBlur(image_mask,(self.kernalSize,self.kernalSize),0, 0)
        # invert the mask so we block out the foreground 
        mask = cv2.bitwise_not(image_mask)
        mask = cv2.merge((mask,mask,mask))
        image_with_mask = cv2.bitwise_and(self.workingImage, mask)
        return image_with_mask

    def threshold(self, upper, lower):
        # create the lower mask - we're adj the lower range in case
        # of a scenerio like upper = [10, 245, 20] & lower = [245, 215, 235]
        # lower should instead be [0, 215, 0]
        lower_adj = []
        for i in range(len(upper)):
            lower_adj.append(0 if upper[i] < lower[i] else lower[i])
        lower_adj = np.array(lower_adj)
        mask1 = cv2.inRange(self.HSVImage, lower_adj, upper)

        # create upper mask with same logic as above
        upper_adj = [180 if upper[0] < lower[0] else upper[0]]
        for i in range(1, len(upper)):
            upper_adj.append(255 if upper[i] < lower[i] else upper[i])
        upper_adj = np.array(upper_adj)
        mask2 = cv2.inRange(self.HSVImage, lower, upper_adj)

        # Generating the final mask to detect the color (just oring the masks together)
        return mask1 + mask2

    def update_image(self):
        # threshold the image and create a mask
        mask = self.threshold(self.upper, self.lower)
        # combine background and foreground images
        final_image = self.remove_background(mask)
        # show the final image
        cv2.imshow(self.windowName, final_image)

    def combine_images(self):
        image_mask = self.threshold(self.upper, self.lower)
        image_with_mask = self.remove_background(image_mask)
        # blur the mask
        image_mask = cv2.GaussianBlur(image_mask,(self.kernalSize,self.kernalSize),0, 0)
        # mask out the foreground
        mask = cv2.merge((image_mask,image_mask,image_mask))
        background_with_mask = cv2.bitwise_and(self.background, mask)
        # combine the images and convert back to BGR
        final_image = background_with_mask + image_with_mask
        cv2.imshow('final video', final_image)

    def green_spill_removal(self):
        # split image and find max of red and green channels
        b, g, r = cv2.split(self.image)
        b_temp = np.copy(b)
        r_temp = np.copy(r)
        b_temp[r >= b] = 0
        r_temp[b > r] = 0
        max_b_r = b_temp + r_temp
        # scale the green image based off selected value
        g_max = g > max_b_r
        scaled_g = g * self.greenSpillScale
        g[g_max] = scaled_g[g_max]
        g = g.astype(np.uint8)
        self.workingImage = cv2.merge([b, g, r])

    def green_screen_frame(self, image):
        self.image = image
        self.HSVImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.green_spill_removal()
        self.combine_images()

    def add_trackbars(self):
        # Create Trackbar to filter color range
        cv2.createTrackbar(
            "Color Range",
            self.windowName,
            20,
            int(255 / 2),
            self.trackbar_callback_color
        )
        # Create Trackbar to smooth mask
        cv2.createTrackbar(
            "Mask Blurring",
            self.windowName,
            5,
            25,
            self.trackbar_callback_guassian_blur_mask
        )
        # Create Trackbar to remove green spill over
        cv2.createTrackbar(
            "Remove Green Spill",
            self.windowName,
            100,
            100,
            self.trackbar_callback_green_spill_removal
        )

    # Callback functions
    def trackbar_callback_color(self, val):
        upper, lower = self.color_selector(val)
        self.upper = upper
        self.lower = lower
        self.update_image()

    def trackbar_callback_guassian_blur_mask(self, kernalSize):
        # kernal size has to be odd
        self.kernalSize = kernalSize if kernalSize % 2 == 1 else kernalSize + 1
        self.update_image()
    
    def trackbar_callback_green_spill_removal(self, scale):
        self.greenSpillScale = scale / 100.0
        self.green_spill_removal()
        self.update_image()

         

# image is too large for my screen so i need to resize it
def resize_image(image):
    scale_percent = 60  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

cap = cv2.VideoCapture("./videos/greenscreen-demo.mp4")
filename = "./images/background.jpg"
background = cv2.imread(filename, cv2.IMREAD_COLOR)
background = resize_image(background)

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")


valid, img = cap.read()
img = resize_image(img)
greenScreenKeying = GreenScreenKeying("image", img, background)

# wait for the user to be done editing 
print('Press d when done editing to play the video')
while True: 
    k = cv2.waitKey(1)
    if k == ord('d') or k == 27: # esc to end edting and skip editing
        break

# play the video 
while cap.isOpened() and k != 27:
    valid, img = cap.read()
    k = cv2.waitKey(25)
    if valid:
        img = resize_image(img)
        greenScreenKeying.green_screen_frame(img)
    if k == 27: # esc key
        break

cap.release()
cv2.destroyAllWindows()

