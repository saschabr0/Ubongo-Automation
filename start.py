from configuration import *
from CardDetector import *
from TileDetector import *
from UbongoSolver import *
from Vizualizer import *
from utils import autoscale
import time
import numpy as np

def main():
    if USE_CAMERA:
        cap = cv2.VideoCapture()
    elif VIDEO_INPUT:
        cap = cv2.VideoCapture(VIDEOPATH)

    zoomFactor = 0.9  # start zoom factor for vizualisation
    pyrfactor = 2  # factor for data reduction
    levels = 16  # number of thresholds for binarization
    autoscaling = True

    cardDetector = CardDetector(pyrfactor, levels)
    tileDetector = TileDetector(visualize = True)
    ubongoSolver = UbongoSolver()
    visualizer = Vizualizer()

    while True:

        if USE_CAMERA:
            if not cap.isOpened():
                cap.open(IMAGE_AQUISITION_CAMERA_ID)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_AQUISITION_FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_AQUISITION_FRAME_HEIGHT)

            cap.grab()
            reti, image = cap.retrieve()

            if not reti:
                continue

            # crop image
            image = image[:, 110:image.shape[1] - 130]

        elif VIDEO_INPUT:
            reti, image = cap.read()
            if not reti:
                break

        else:
            width = 800
            height = 600
            dim = (width, height)
            image = cv2.imread(IMAGEPATH)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            if image is None:
                print()
                print("WARNING: Image could not be loaded")
                print("Application has ended")
                break

        if autoscaling:
            image = autoscale(image)

        start = time.time()

        ubongoCards = cardDetector.compute(image.copy())
        tileContours = tileDetector.compute(image.copy())
        card_cnts = [card.contour for card in ubongoCards]
        ubongoTiles = tileDetector.removeCards(card_cnts, tileContours)

        for i, card in enumerate(ubongoCards):

            if card.playFieldPattern is None or len(card.blocks) == 0:
                continue

            card.isValid, card.blockPositions = ubongoSolver.findUbongoSolution(card.playFieldPattern, card.blocks)

        t = (time.time() - start) * 1e3

        vizImage = visualizer.createVizImage(image.copy(), ubongoCards)

        cv2.putText(vizImage, "%.3f ms" % t, (image.shape[1] - 120, image.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 1.25,
                    (255, 255, 255), 1)
        vizImage = cv2.resize(vizImage, (int(vizImage.shape[1] * zoomFactor), int(vizImage.shape[0] * zoomFactor)))


        cv2.imshow("Viz", vizImage)

        keyPress = chr(cv2.waitKey(1) & 255)
        if keyPress == 'q':
            break
        elif keyPress == 's' and USE_CAMERA:
            cv2.imwrite('data/img_viz.png', vizImage)
        elif keyPress == '+' and zoomFactor < 2:
            zoomFactor = zoomFactor + 0.05
        elif keyPress == '-' and zoomFactor > 0.1:
            zoomFactor = zoomFactor - 0.05
        elif keyPress == 'a':
            autoscaling = False if autoscaling else True

    if USE_CAMERA or VIDEO_INPUT:
        cap.release()


if __name__ == '__main__':
    main()
