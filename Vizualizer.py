import cv2
import numpy as np
from configuration import *

class Vizualizer(object):

    def __init__(self):
        self.offset = 10
        self.transparency = 1


    def createVizImage(self, image, ubongoCards):
        y, x, c = image.shape

        columnWidth = 140

        for card in ubongoCards:
            if len(card.contour) > 0:
                cv2.drawContours(image, [card.contour], -1, (0, 255, 0), 2)

        vizImage = np.ones((y, x + columnWidth * len(ubongoCards) + self.offset * (len(ubongoCards)+1), c), np.uint8) * 120
        vizImage[:y, :x] = image[:, :]

        for i, card in enumerate(ubongoCards):
            vizInfo = vizImage[:, x + self.offset + (columnWidth + self.offset) * i :x + (self.offset + columnWidth) *(i+1)]
            vizInfo = self.fillVizInfo(vizInfo, card)

        return vizImage

    def fillVizInfo(self, vizInfo, card):

        if card.image is None:
            return

        resizeFactor = 1
        if vizInfo.shape[1] < card.image.shape[1]:
            resizeFactor = vizInfo.shape[1] / card.image.shape[1]

        elif vizInfo.shape[0] < card.image.shape[0]:
            resizeFactor = vizInfo.shape[0] / card.image.shape[0]

        card.image = cv2.resize(card.image, (int(card.image.shape[1] * resizeFactor), int(card.image.shape[0] * resizeFactor)))
        y, x, c = card.image.shape

        firstRow = vizInfo[self.offset: self.offset + y, :x]
        secondRow = vizInfo[y + 2 * self.offset: , :x]

        self.fillFirstRow(firstRow, card, resizeFactor)

        if len(card.blocks) == 0 or len(card.blocks) > 3:
            return

        self.fillSecondRow(secondRow, card)


    def fillFirstRow(self, image, card, resizeFactor):

        image[:, :] = card.image[:, :]

        if len(card.playFieldContour) <= 0:
            return

        card.playFieldContour = (card.playFieldContour * resizeFactor).astype(np.uint8)

        if card.isValid and len(card.blockPositions) > 0 and len(card.blockPositions) <=3:
            # draw solution
            overlay = image.copy()

            for i, block in enumerate(card.blockPositions[::-1]):
                self.drawPlayFieldPattern(overlay, card.playFieldContour, block, BLOCKCOLORS[i])

            image[:, :] = cv2.addWeighted(overlay, self.transparency, image[:, :], 1 - self.transparency, 0)

        else:
            # draw only pattern
            overlay = image.copy()
            self.drawPlayFieldPattern(overlay, card.playFieldContour, card.playFieldPattern, (0,255,0))
            image[:, :] = cv2.addWeighted(overlay, self.transparency-0.1, image[:, :], 1 - self.transparency+0.1, 0)


    def fillSecondRow(self, image, card):
        y_offset = 0
        for i, block in enumerate(card.blocks):
            self.drawBlockPattern(image, y_offset, block, BLOCKCOLORS[i])
            y_offset += block.shape[0] + 1


    def drawPlayFieldPattern(self, image, contour, pattern, color):

        [[xmin, ymin]] = contour.min(axis=0)
        [[xmax, ymax]] = contour.max(axis=0)

        fieldSize = int((xmax-xmin) / (pattern.shape[1] +0.01))

        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                if pattern[i, j]:
                    cv2.rectangle(image, (xmin + fieldSize*j, ymin + fieldSize*i), (xmin+fieldSize + fieldSize*j , ymin+fieldSize + fieldSize*i), color, -1)


    def drawBlockPattern(self, image, y_offset, pattern, color):

        fieldSize = int(image.shape[1] / 6.5)

        for i in range(pattern.shape[0]):
            for j in range(pattern.shape[1]):
                if pattern[i, j]:
                    cv2.rectangle(image, (int(fieldSize/2) + fieldSize*j, int(fieldSize) + y_offset * fieldSize + fieldSize *i), (int(fieldSize/2) + fieldSize + fieldSize*j, int(fieldSize) + y_offset*fieldSize + fieldSize + fieldSize*i), color, -1)
