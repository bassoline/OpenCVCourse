import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.ion()

matplotlib.rcParams["figure.figsize"] = (10.0, 10.0)
matplotlib.rcParams["image.cmap"] = "gray"

kernelSize = 3

# workoncv-4.1.0
# ipython


class BlemishRemover:
    def __init__(self, windowname, image, originalImage):
        self.windowname = windowname
        self.image = image
        self.oringinalImage = originalImage
        self.clicks = []
        self.clickMap = {}
        self.padded = image
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)

    # show the image
    def show(self):
        cv2.imshow(self.windowname, self.image)

    # onMouse function for handling clicking
    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.removeBlemish(x, y)

    def sobelGradient(self, subsection):
        # take grayscale of image
        grey = cv2.cvtColor(subsection, cv2.COLOR_BGR2GRAY)
        # smooth image
        smoothed = cv2.GaussianBlur(grey, (3, 3), 0, 0)
        # calculate sobel gradient over subsection
        sobel = cv2.Sobel(smoothed, cv2.CV_32F, 1, 1)
        # sum up pixels so we can compare
        sobelSum = abs(np.mean(sobel))
        return sobelSum

    def colorDiff(self, subsection1, subsection2):
        b1, g1, r1 = cv2.split(subsection1)
        b2, g2, r2 = cv2.split(subsection2)
        bDiff = self.normalizeColorDiff(b1, b2)
        gDiff = self.normalizeColorDiff(g1, g2)
        rDiff = self.normalizeColorDiff(r1, r2)
        return bDiff + gDiff + rDiff

    def normalizeColorDiff(self, colorChannelA, colorChannelB):
        # dividing by 2000 to bring in the range of sobel diff
        return abs(np.sum(colorChannelA - colorChannelB)) / 2000

    def getSubSection(self, x, y):
        return self.image[y - 15 : y + 15, x - 15 : x + 15]

    def seemlessClone(self, subx, suby, mouseX, mouseY):
        subSection = self.getSubSection(subx, suby)
        mask = 255 * np.ones(subSection.shape, subSection.dtype)
        center = (mouseX, mouseY)
        self.image = cv2.seamlessClone(
            subSection, self.image, mask, center, cv2.NORMAL_CLONE
        )

    def createPaddedImage(self):
        row, col = self.image.shape[:2]
        paddedImage = np.zeros((row + 30, col + 30, 3), dtype="uint8")
        paddedImage[15 : row + 15, 15 : col + 15] = self.image
        self.padded = paddedImage

    def getBestRegion(self, x, y):
        # create padded image to do calculations on
        self.createPaddedImage()
        # calculate regions 15 pixels away
        colorDiffMap = {}
        sobelMap = {}
        # take 9 square regions around clicked pixel
        # and find their gradients and sum of color differences
        clickedSubSection = self.getSubSection(x, y)

        for r in range(-1, 2, 1):
            for c in range(-1, 2, 1):
                newX = x + 15 * r
                newY = y + 15 * c
                newSubsection = self.getSubSection(newX, newY)
                colorDiffMap[(newX, newY)] = self.colorDiff(
                    clickedSubSection, newSubsection
                )
                sobelMap[(newX, newY)] = self.sobelGradient(newSubsection)

        # clicked area gets min of avg and median for color diff
        # (if it has the smallest graident it's possible that the picture doesn't change)
        bestZone = (x, y)
        avgColorDiff = sum(colorDiffMap.values()) / 8
        medianColorDiff = np.median(list(colorDiffMap.values()))
        colorDiffMap[bestZone] = (
            avgColorDiff if avgColorDiff < medianColorDiff else medianColorDiff
        )
        lowestSum = colorDiffMap[bestZone] + sobelMap[bestZone]

        for key, value in colorDiffMap.items():
            tempSum = value + sobelMap[bestZone]
            if tempSum < lowestSum:
                lowestSum = tempSum
                bestZone = key

        print("clicked", (x, y))
        print("best zone", bestZone)
        return bestZone

    def removeBlemish(self, x, y):
        # save clicked region for undoing
        self.clicks.append((x, y))
        self.clickMap[(x, y)] = self.getSubSection(x, y)
        # find the smoothest region
        bestZone = self.getBestRegion(x, y)
        # seemless clone and show image
        self.seemlessClone(bestZone[0], bestZone[1], x, y)
        self.show()

    def undo(self):
        x, y = self.clicks.pop()
        self.image[y - 15 : y + 15, x - 15 : x + 15] = self.clickMap[(x, y)]
        del self.clickMap[(x, y)]
        self.show()


# Read image
# img = cv2.imread("./images/blemish.png", cv2.IMREAD_COLOR)
img = cv2.imread("./images/IMG_2437.jpeg", cv2.IMREAD_COLOR)
before = img.copy()

# Create blemish remover instance
blemishRemover = BlemishRemover("image", img, before)

# capture the pixel the user clicked on
while True:
    ch = cv2.waitKey()
    if ch == 27:  # esc key
        print("closing")
        break
    # undo click
    if ch == ord("u"):
        print("undoing")
        blemishRemover.undo()
    # restart click
    if ch == ord("r"):
        print("restarting")
        blemishRemover.image = before
        blemishRemover.show()

cv2.destroyAllWindows()
