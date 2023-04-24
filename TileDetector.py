import cv2
import numpy as np
from Tile import *

class TileDetector:
	def __init__(self, visualize = False):
		self.visualize = visualize

	def dist(self, pt1, pt2):
		return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

	def getAngle(self, cnt, centroid):
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)

		## Aus Konturwert wird das kleinstmögliche Rechteck erstellt

		center = (int(rect[0][0]), int(rect[0][1]))
		width = int(rect[1][0])
		height = int(rect[1][1])
		angle = int(rect[2])

		if width < height:
			angle = 90 - angle

		else:
			angle = -angle


		if self.visualize:
			cv2.drawContours(self.original,[box],0,(0,0,255),2)
			label = "Angle: " + str(angle) + " | " + "x: " + str(int(centroid[0]))+ " y: " + str(int(centroid[1])) 
			
			# textbox = cv2.rectangle(self.original, (center[0]-35, center[1]-25), 
			# 	(center[0] + 295, center[1] + 10), (255,255,255), -1)
			
			cv2.putText(self.original, label, (center[0]-50, center[1]), 
						cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

		# Winkel, X- und Y-Koordinaten werden anhand der 0 | 0 Position des Bildes berechnet

		return angle

	def removeCards(self, card_cnts, tileCnts):
		centroids_tiles = [np.mean(cnt.squeeze(), axis = 0) for cnt in tileCnts]
		centroids_cards = [np.mean(cnt.squeeze(), axis = 0) for cnt in card_cnts]

		## Liste mit Schwerpunkte aller Teile und Karten

		newTileCnts = []
		self.allTiles = []
		for i, ct in enumerate(centroids_tiles):
			valid = True
			for cc in centroids_cards:
				d = self.dist(ct, cc)
				if d<50:
					valid = False
			if valid:
				newTileCnts.append(tileCnts[i])
				tile = Tile()
				tile.contour = tileCnts[i]
				tile.centroid = ct
				tile.angle = self.getAngle(tileCnts[i], ct)
				self.allTiles.append(tile)

		##Es wird auf neue Teile untersucht, die einen Mindestabstand zueinander haben; Abspeichern in Liste

		if self.visualize:
			for cnt in newTileCnts:
				cv2.drawContours(self.original, [cnt], -1, 255, 1)

			# self.original[20][50] = (255, 255, 255)

			cv2.imshow("Tile Detection", self.original)

		return self.allTiles

		##Konturen werden gezeichnet

	def compute(self, image):
		self.original = image.copy()

		image = self.findAndDrawContours(image)
		image, tileCnts = self.findFinalContours(image)

		cv2.imshow("Tile Detection", image)

		return tileCnts

		##Ausgabe des Bildes mit Konturen

	def findFinalContours(self, image):
		low = (254, 0, 0)
		high = (255, 0, 0)

		mask = cv2.inRange(image, low, high)
		cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		valid_cnts = []

		for cnt in cnts:
			area = cv2.contourArea(cnt)

			arcLength = cv2.arcLength(cnt, True)
			approxCnt = cv2.approxPolyDP(cnt, 0.03 * arcLength, True)

			if area>100:
				if len(approxCnt)>=4:
					cv2.drawContours(image, [approxCnt], -1, 255, 1)
					valid_cnts.append(approxCnt)

					##Teil hat 4 oder mehr approximierte Konturseiten

			##Teil groß genug --> validiert

		return image, valid_cnts

	def findAndDrawContours(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (7, 7), 0)

		edgeMask = cv2.Canny(blurred, 40, 140)
		# cv2.imshow("Canny", edgeMask)
		edgeMask = cv2.dilate(edgeMask, None, iterations=1)
		edgeMask = cv2.erode(edgeMask, None, iterations=1)

		cnts, hierarchy = cv2.findContours(edgeMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		valid_cnts = []

		##Canny Edge Detector und Dilation und Erosion

		for cnt in cnts:
			area = cv2.contourArea(cnt)

			arcLength = cv2.arcLength(cnt, True)
			approxCnt = cv2.approxPolyDP(cnt, 0.03 * arcLength, True)

			if area>100:
				if len(approxCnt)>=4:
					cv2.drawContours(image, [approxCnt], -1, 255, -1)
					valid_cnts.append(approxCnt)

		return image
