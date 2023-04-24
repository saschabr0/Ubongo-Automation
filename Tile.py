
class Tile:
	def __init__(self):
		self.contour = None									##Konturwert
		self.centroid = None								##Schwerpunkt
		self.angle = None									##Winkel

	def getPosition(self):
		return self.centroid[0], self.centroid[1]			##X und Y Position

	def __str__(self):
		desc = "This is a Tile"
		return desc