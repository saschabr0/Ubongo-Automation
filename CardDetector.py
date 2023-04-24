import cv2
import numpy as np
from UbongoCard import *
from utils import autoscale


class CardDetector(object):

    # Size: card: 9.1cm x 5.9cm, field = 1.24cm x 1.24 cm

    def __init__(self, pyrfactor=2, levels=16):
        self.pyrfactor = pyrfactor
        self.levels = levels
        self.originImage = None


    def compute(self, image):

        self.originImage = image

        for _ in range(self.pyrfactor-1):
            image = cv2.pyrDown(image)

        self.wImage = self.getWhitenessImage(image)
        self.contourList = self.findContours(self.wImage)

        self.cardContours = self.findCardinContours(self.contourList)
        self.cardContours = self.mergeCardContours(self.cardContours)

        if len(self.cardContours) == 0:
            return []

        self.cards = []
        for i, cnt in enumerate(self.cardContours):
            card = UbongoCard()

            card.contour = self.refindCard(cnt, image)
            card.image = self.doAffineTransform(card.contour)

            if card.image is None:
                continue

            card.image = self.correctIllumination(card.image)
            card.image = self.correctCardOrientation(card.image)

            card.playFieldPattern, card.playFieldContour = self.detectPlayField(card.image.copy())
            card.blocks = self.detectBlocks(card.image.copy())

            self.cards.append(card)

        self.cards = self.sortCards(self.cards)

        return self.cards


    def getWhitenessImage(self, image):

        w = image[...].min(axis=2)

        return np.array(w, np.uint8)


    def findContours(self, image):

        contourList = []
        for i in np.linspace(0, 255, self.levels+1, endpoint=False)[1:]:
            th, bin = cv2.threshold(image, i, 255, cv2.THRESH_BINARY)

            ks = int(32 * 2**-self.pyrfactor) | 1
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            m = cv2.morphologyEx(bin, cv2.MORPH_OPEN, se)

            contours, hierarchy = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contourList.extend(contours)

        return contourList


    def findCardinContours(self, contourList):
        # print()
        cardContours = []
        for cnt in contourList:
            area = cv2.contourArea(cnt)
            minArea = 4000 * 2**-self.pyrfactor
            minArea = 5000
            maxArea = 180000 * 2**-self.pyrfactor
            # print(minArea, area, maxArea)

            # 1st criterion: size
            if minArea <= area <= maxArea:
                arcLength = cv2.arcLength(cnt, True)
                approxCnt = cv2.approxPolyDP(cnt, 0.03 * arcLength, True)

                # 2nd criterion: four corners
                if len(approxCnt) == 4:
                    distList = list()
                    for i in range(-1, 3):
                        dist = abs((approxCnt[i][0][0] - approxCnt[i + 1][0][0]) ** 2 + (approxCnt[i][0][1] - approxCnt[i + 1][0][1]) ** 2)
                        distList.append(dist)

                    # 3rd criterion: rectangular (top and bottom as well as left and right side must be the same length)
                    if abs(distList[0] - distList[2]) <= (48**2 * 2**-self.pyrfactor) and abs(distList[1] - distList[3]) <= (48**2 * 2**-self.pyrfactor):
                        # UbongoCards are not square
                        if abs(distList[0] - distList[1]) > 48 * 2**-self.pyrfactor:
                            cardContours.append(approxCnt)

        return cardContours


    def mergeCardContours(self, cardContours):

        if len(cardContours) == 0:
            return []

        centroids = [cnt.mean(axis=0)[0] for cnt in cardContours]

        # sort the contours in groups
        handledCntIdx = list()
        groupedCntIdx = list()
        for i, c in enumerate(cardContours):
            if i in handledCntIdx:
                continue
            handledCntIdx.append(i)
            groupedCntIdx.append([i])
            cPoint = centroids[i]

            for j, d in enumerate(cardContours):
                if j in handledCntIdx:
                    continue
                if self.doesPointsMatch(cPoint, centroids[j], int(80 * 2**-self.pyrfactor), int(80 * 2**-self.pyrfactor)):
                    handledCntIdx.append(j)
                    groupedCntIdx[-1].append(j)


        # merge all cards of a group into one contour
        newCardContours = []
        for i, idxgrouplist in enumerate(groupedCntIdx):
            newCardContours.append(cardContours[idxgrouplist[0]])
            for idx in idxgrouplist[1:]:
                for j, p in enumerate(cardContours[idx]):
                    try:
                        newCardContours[i][j] = (newCardContours[i][j] + p) / 2
                    except:
                        # print("error")
                        pass

        return newCardContours


    def doesPointsMatch(self, cPoint, centroid, areaWidth, areaHeight):
        return (abs(centroid[0]-cPoint[0]) <= areaWidth and abs(centroid[1]-cPoint[1]) <= areaHeight)


    def refindCard(self, cnt, image):

        if len(cnt) == 0:
            return []

        # get contour size plus offset
        [[xmin, ymin]] = cnt.min(axis=0) - int(48 * 2**-self.pyrfactor)
        [[xmax, ymax]] = cnt.max(axis=0) + int(48 * 2**-self.pyrfactor)

        if xmin < 0 or ymin < 0 or xmax > image.shape[1] or ymax > image.shape[0]:
            return []

        w = self.wImage[ymin:ymax, xmin:xmax]

        th, bin = cv2.threshold(w, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ks = int(32 * 2 ** -self.pyrfactor) | 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

        m = 255 - cv2.morphologyEx(bin, cv2.MORPH_OPEN, se)
        newContours, newHierarchy = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        minContourArea = (m.shape[0] * m.shape[1]) * 0.15

        if len(newContours) > 1:
            # find the contour that is closest to the image center
            [icx, icy] = [int(w.shape[0]/2), int(w.shape[1]/2)]
            distList = []
            for cnt in newContours:
                if cv2.contourArea(cnt) < minContourArea:
                    continue
                [[cx, cy]] = cnt.mean(axis=0)
                distList.append(abs(icx-cx)**2 + abs(icy-cy)**2)

            if (len(distList)) > 0:
                cardContour = newContours[np.argmin(distList)]

            else:
                return []

        elif len(newContours) == 1 and cv2.contourArea(newContours[0]) >= minContourArea:
            cardContour = newContours[0]
        else:
            return []

        # convert contour to original image size
        cardContour = (cardContour + [xmin, ymin]) * 2**(self.pyrfactor-1)

        return cardContour



    def doAffineTransform(self, cardContour):

        if len(cardContour) == 0:
            return None

        rect = cv2.minAreaRect(cardContour)
        box = cv2.boxPoints(rect)

        # find out where the first point is
        dist_01 = ((box[1] - box[0]) ** 2).sum()
        dist_03 = ((box[3] - box[0]) ** 2).sum()

        if dist_03 > dist_01:
            src_p0 = box[0]
            src_p1 = box[1]
            src_p2 = box[2]
            cols = int(rect[1][1])
            rows = int(rect[1][0])
        else:
            src_p0 = box[3]
            src_p1 = box[0]
            src_p2 = box[1]
            cols = int(rect[1][0])
            rows = int(rect[1][1])

        src_pt = np.float32([src_p0, src_p1, src_p2])
        dst_pt = np.float32(
            [[min(rect[1][1], rect[1][0]), max(rect[1][1], rect[1][0])], [0, max(rect[1][1], rect[1][0])], [0, 0]])

        M = cv2.getAffineTransform(src_pt, dst_pt)
        dst = cv2.warpAffine(self.originImage, M, (cols, rows))

        return dst


    def correctCardOrientation(self, image):

        if image is None:
            return None

        w = image[...].min(axis=2)

        # calculate the line averages
        avg = cv2.reduce(w, 1, cv2.REDUCE_AVG)

        upper = avg[0:int(avg.shape[0]/3)]
        lower = avg[int(avg.shape[0] * 2/3):]

        if lower.mean() > upper.mean():
            image = cv2.flip(image, 0)
            image = cv2.flip(image, 1)

        return image


    def calcIlluminationCourse(self, val1, val2, s, scale=1):

        reffac = 1 if val1 < val2 else -1
        dval = abs(val1-val2)
        a = np.array(np.linspace(0, dval, s) * np.linspace(0, 1, s)**2 * scale, np.uint8)[::reffac]

        return a


    def correctIllumination(self, image):

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l = lab[..., 0]

        y, x = l.shape[:2]
        yh = y >> 1
        xh = x >> 1
        r = 10
        s = 5

        illu_hor = self.calcIlluminationCourse(l[yh-s:yh+s, r-s:r+s].mean(), l[yh-s:yh+s, (x-r)-s:(x-r)+s].mean(), x)
        illu_ver = self.calcIlluminationCourse(l[r-s:r+s, xh-s:xh+s].mean(), l[(y-r)-s:(y-r)+s, xh-s:xh+s].mean(), y)

        # resize to image size
        illu_hor = np.repeat(illu_hor, y).reshape(x, -1).T
        illu_ver = np.repeat(illu_ver, x).reshape(y, -1)

        l[...] = cv2.subtract(l, illu_ver)[...]
        l[...] = cv2.subtract(l, illu_hor)[...]

        # autoscaling
        l_c = np.zeros((l.shape[0], l.shape[1], 1))
        l_c[:, :, 0] = l[:, :]
        l_c = autoscale(l_c, True, 0.01, 0.99)
        l[...] = l_c[:, :, 0]

        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return image


    def detectPlayField(self, image):

        if image is None:
            return None, []

        w = image[...].min(axis=2)

        th, bin = cv2.threshold(w, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ks = int(25 * 2 ** -self.pyrfactor) | 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

        cardBin = cv2.morphologyEx(bin, cv2.MORPH_OPEN, se)
        cardBin = cv2.morphologyEx(cardBin, cv2.MORPH_CLOSE, se)

        contours, hierarchy = cv2.findContours(cardBin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 1:
            cntArea = []
            for cnt in contours:
                cntArea.append(cv2.contourArea(cnt))

            playAreaContour = contours[np.argmax(cntArea)]

        elif len(contours) == 1:
            playAreaContour = contours[0]

        else:
            return None, []

        playAreaPattern = self.getPatternFromContour(cardBin, playAreaContour, (0.137, 0.213))

        if playAreaPattern is None or playAreaPattern.shape[0] > 5 or playAreaPattern.shape[1] > 4:
            return None, []

        return playAreaPattern, playAreaContour


    def detectBlocks(self, image):

        if image is None:
            return []

        lowerPart = image[int(image.shape[0] * 3 / 4):]

        c = cv2.subtract(lowerPart[...].max(axis=2), lowerPart[...].min(axis=2))

        th, bin = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ks = int(13 * 2 ** -self.pyrfactor) | 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))

        bin = cv2.morphologyEx(bin, cv2.MORPH_OPEN, se)

        contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        blockList = []
        for cnt in contours:
            [[w,h]] = cnt.max(axis=0)-cnt.min(axis=0)
            if h >= (lowerPart.shape[0] * 0.10) or w >= (lowerPart.shape[1] * 0.10):
                blockList.append(cnt)

        blockList = self.sortBlocks(blockList)

        blocks = []
        for cnt in blockList:
            blockPattern = self.getPatternFromContour(bin, cnt, (0.233, 0.086))
            if blockPattern is None or blockPattern.shape[0] > 3 or blockPattern.shape[1] > 4:
                continue
            blocks.append(blockPattern)

        return blocks


    def getPatternFromContour(self, binary, contour, relativSize):

        [[xmin, ymin]] = np.min(contour, axis=0)
        [[xmax, ymax]] = np.max(contour, axis=0)
        height = ymax - ymin
        width = xmax - xmin

        # calculate number of rows and columns of the pattern
        size = (int(np.round(height/binary.shape[0]/relativSize[0])), int(np.round(width/binary.shape[1]/relativSize[1])))

        if any(s == 0 for s in size) or any(s > 5 for s in size):
            return None

        # calculate pixel size of one field
        fieldSize = int(height / size[0] + 1e-6)
        if fieldSize <= 0:
            return None

        # get value of the fields (0 or 1)
        pattern = (binary[ymin + int(fieldSize / 2):ymax:fieldSize, xmin + int(fieldSize / 2):xmax:fieldSize]) / 255.

        return pattern


    def sortBlocks(self, blocks):

        blockPos = [b.min(axis=0)[0][0] for b in blocks]

        # sort the blocks from left to right
        sortedBlockList = []
        for i in range(len(blocks)):
            sortedBlockList.append(blocks[np.argmin(blockPos)])
            blockPos[np.argmin(blockPos)] = np.inf

        return sortedBlockList


    def sortCards(self, cards):

        cardPos = []
        for c in cards:
            mi = c.contour.min(axis=0)[0]
            cardPos.append(mi[0]**2 + mi[1]**2)

        # sort cards by distance to Point (0,0)
        sortedCardList = []
        for i in range(len(cards)):
            sortedCardList.append(cards[np.argmin(cardPos)])
            cardPos[np.argmin(cardPos)] = np.inf

        return sortedCardList
