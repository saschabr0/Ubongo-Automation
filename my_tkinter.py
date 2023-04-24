import tkinter as tk
import tkinter.font as tkFont
import numpy
import numpy as np
from PIL import Image, ImageTk
import cv2 as cv
from CardDetector import *
from TileDetector import *
from UbongoSolver import *
from Vizualizer import *
import colorfiltering
import time
import globals
import threading
from imutils.perspective import four_point_transform
import utils
import os
import precise

green = (0, 255, 0)
cap = cv.VideoCapture(0)
image_preview = Image.open("data/preview.png")
image_preview = image_preview.resize((540, 360), Image.Resampling.LANCZOS)
image_logo = Image.open("data/logo.png")


class App:
    def __init__(self, root):
        root.title("Ubongo Automation")
        width = 1280
        height = 800
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        GLabel_91 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=38)
        GLabel_91["font"] = ft
        GLabel_91["fg"] = "firebrick1"
        GLabel_91["justify"] = "center"
        GLabel_91["text"] = "Ubongo Automation"
        GLabel_91.place(x=100, y=20, width=500, height=62)

        self.logo = ImageTk.PhotoImage(image_logo)
        self.left_logo = tk.Label(root, image=self.logo, width=60, height=40)
        self.left_logo.place(x=30, y=30, width=60, height=40)
        self.right_logo = tk.Label(root, image=self.logo, width=60, height=40)
        self.right_logo.place(x=620, y=30, width=60, height=40)

        GLabel_83 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=18)
        GLabel_83["font"] = ft
        GLabel_83["fg"] = "#333333"
        GLabel_83["justify"] = "left"
        GLabel_83["text"] = "Insert filepath or leave empty for Live-Cam"
        GLabel_83.place(x=50, y=100, width=500, height=40)

        self.entry_path = tk.Entry(root)
        self.entry_path["borderwidth"] = "1px"
        ft = tkFont.Font(family='Times', size=10)
        self.entry_path["font"] = ft
        self.entry_path["fg"] = "#333333"
        self.entry_path["justify"] = "center"
        self.entry_path["text"] = ""
        self.entry_path.place(x=70, y=140, width=500, height=30)

        lbl_path = tk.Label(root)
        ft = tkFont.Font(family='Times', size=10)
        lbl_path["font"] = ft
        lbl_path["fg"] = "#333333"
        lbl_path["justify"] = "center"
        lbl_path["text"] = "file path:"
        lbl_path.place(x=0, y=140, width=70, height=25)

        lbl_start_pos = tk.Label(root)
        ft = tkFont.Font(family='Times', size=20)
        lbl_start_pos["font"] = ft
        lbl_start_pos["fg"] = "gray20"
        lbl_start_pos["justify"] = "center"
        lbl_start_pos["text"] = "Start Position:"
        lbl_start_pos.place(x=190, y=200, width=300, height=75)

        lbl_start_img_coordinates = tk.Label(root)
        ft = tkFont.Font(family='Times', size=18)
        lbl_start_img_coordinates["font"] = ft
        lbl_start_img_coordinates["fg"] = "gray20"
        lbl_start_img_coordinates["justify"] = "center"
        lbl_start_img_coordinates["text"] = "Image Coordinates:"
        lbl_start_img_coordinates.place(x=50, y=250, width=300, height=75)

        lbl_start_rob_coordinates = tk.Label(root)
        ft = tkFont.Font(family='Times', size=18)
        lbl_start_rob_coordinates["font"] = ft
        lbl_start_rob_coordinates["fg"] = "gray20"
        lbl_start_rob_coordinates["justify"] = "center"
        lbl_start_rob_coordinates["text"] = "Robot Coordinates:"
        lbl_start_rob_coordinates.place(x=350, y=250, width=300, height=75)

        self.lbl_start_img_t1 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        self.lbl_start_img_t1["font"] = ft
        self.lbl_start_img_t1["fg"] = "black"
        self.lbl_start_img_t1["bg"] = "lemon chiffon"
        self.lbl_start_img_t1["justify"] = "center"
        self.lbl_start_img_t1["text"] = "Tile 1"
        self.lbl_start_img_t1.place(x=20, y=320, width=150, height=100)

        self.lbl_start_img_t2 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        self.lbl_start_img_t2["font"] = ft
        self.lbl_start_img_t2["fg"] = "black"
        self.lbl_start_img_t2["bg"] = "LightBlue1"
        self.lbl_start_img_t2["justify"] = "center"
        self.lbl_start_img_t2["text"] = "Tile 2"
        self.lbl_start_img_t2.place(x=180, y=320, width=150, height=100)

        self.lbl_start_rob_t1 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        self.lbl_start_rob_t1["font"] = ft
        self.lbl_start_rob_t1["fg"] = "black"
        self.lbl_start_rob_t1["bg"] = "lemon chiffon"
        self.lbl_start_rob_t1["justify"] = "center"
        self.lbl_start_rob_t1["text"] = "Tile 1"
        self.lbl_start_rob_t1.place(x=370, y=320, width=150, height=100)

        self.lbl_start_rob_t2 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        self.lbl_start_rob_t2["font"] = ft
        self.lbl_start_rob_t2["fg"] = "black"
        self.lbl_start_rob_t2["bg"] = "LightBlue1"
        self.lbl_start_rob_t2["justify"] = "center"
        self.lbl_start_rob_t2["text"] = "Tile 2"
        self.lbl_start_rob_t2.place(x=530, y=320, width=150, height=100)

        lbl_rob_coordinates = tk.Label(root)
        ft = tkFont.Font(family='Times', size=20)
        lbl_rob_coordinates["font"] = ft
        lbl_rob_coordinates["fg"] = "gray20"
        lbl_rob_coordinates["justify"] = "center"
        lbl_rob_coordinates["text"] = "End Position:"
        lbl_rob_coordinates.place(x=190, y=450, width=300, height=50)

        lbl_end_img_coordinates = tk.Label(root)
        ft = tkFont.Font(family='Times', size=18)
        lbl_end_img_coordinates["font"] = ft
        lbl_end_img_coordinates["fg"] = "gray20"
        lbl_end_img_coordinates["justify"] = "center"
        lbl_end_img_coordinates["text"] = "Image Coordinates:"
        lbl_end_img_coordinates.place(x=50, y=500, width=300, height=50)

        lbl_end_rob_coordinates = tk.Label(root)
        ft = tkFont.Font(family='Times', size=18)
        lbl_end_rob_coordinates["font"] = ft
        lbl_end_rob_coordinates["fg"] = "gray20"
        lbl_end_rob_coordinates["justify"] = "center"
        lbl_end_rob_coordinates["text"] = "Robot Coordinates:"
        lbl_end_rob_coordinates.place(x=350, y=500, width=300, height=50)

        self.lbl_end_img_t1 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        self.lbl_end_img_t1["font"] = ft
        self.lbl_end_img_t1["fg"] = "black"
        self.lbl_end_img_t1["bg"] = "lemon chiffon"
        self.lbl_end_img_t1["justify"] = "center"
        self.lbl_end_img_t1["text"] = "Tile 1"
        self.lbl_end_img_t1.place(x=20, y=570, width=150, height=100)

        self.lbl_end_img_t2 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        self.lbl_end_img_t2["font"] = ft
        self.lbl_end_img_t2["fg"] = "black"
        self.lbl_end_img_t2["bg"] = "LightBlue1"
        self.lbl_end_img_t2["justify"] = "center"
        self.lbl_end_img_t2["text"] = "Tile 2"
        self.lbl_end_img_t2.place(x=180, y=570, width=150, height=100)

        self.lbl_end_rob_t1 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        self.lbl_end_rob_t1["font"] = ft
        self.lbl_end_rob_t1["fg"] = "black"
        self.lbl_end_rob_t1["bg"] = "lemon chiffon"
        self.lbl_end_rob_t1["justify"] = "center"
        self.lbl_end_rob_t1["text"] = "Tile 1"
        self.lbl_end_rob_t1.place(x=370, y=570, width=150, height=100)

        self.lbl_end_rob_t2 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=14)
        self.lbl_end_rob_t2["font"] = ft
        self.lbl_end_rob_t2["fg"] = "black"
        self.lbl_end_rob_t2["bg"] = "LightBlue1"
        self.lbl_end_rob_t2["justify"] = "center"
        self.lbl_end_rob_t2["text"] = "Tile 2"
        self.lbl_end_rob_t2.place(x=530, y=570, width=150, height=100)

        lbl_origin_robot = tk.Label(root)
        ft = tkFont.Font(family='Times', size=12)
        lbl_origin_robot["font"] = ft
        lbl_origin_robot["fg"] = "black"
        lbl_origin_robot["justify"] = "left"
        lbl_origin_robot["text"] = "Origin of paper in robot coordinates: X = 497 (-); Y = 3 (+); Z = -129 (+)"
        lbl_origin_robot.place(x=100, y=750, width=500, height=50)

        self.lbl_origin_card = tk.Label(root)
        ft = tkFont.Font(family='Times', size=10)
        self.lbl_origin_card["font"] = ft
        self.lbl_origin_card["fg"] = "black"
        self.lbl_origin_card["justify"] = "left"
        self.lbl_origin_card["text"] = "Centroid of the card: "
        self.lbl_origin_card.place(x=100, y=730, width=500, height=30)

        self.lbl_robot_status = tk.Label(root)
        ft = tkFont.Font(family='Times', size=10)
        self.lbl_robot_status["font"] = ft
        self.lbl_robot_status["fg"] = "black"
        self.lbl_robot_status["justify"] = "left"
        self.lbl_robot_status["text"] = "Status of Robot: "
        self.lbl_robot_status.place(x=100, y=700, width=500, height=30)

        self.img_sample = ImageTk.PhotoImage(image_preview)
        self.l_sample = tk.Label(root, image=self.img_sample, width=500, height=250)
        self.l_sample.place(x=700, y=20, width=540, height=360)

        self.img_bbox = ImageTk.PhotoImage(image_preview)
        self.l_bbox = tk.Label(root, image=self.img_bbox, width=500, height=250)
        self.l_bbox.place(x=700, y=400, width=540, height=360)

        self.btn_input = tk.Button(root, text='Get Input', command=self.getInput_clicked, state="active")
        self.btn_input.place(x=585, y=140, width=100, height=25)

        self.btn_getWrap = tk.Button(root, text='Get Wrapped Image', command=self.getWrap_clicked, state="disabled")
        self.btn_getWrap.place(x=100, y=180, width=200, height=25)

        self.btn_getCam = tk.Button(root, text='Detect and Solve', command=self.getCam_clicked, state="disabled")
        self.btn_getCam.place(x=350, y=180, width=200, height=25)

        self.btn_close = tk.Button(root, text="Close", command=root.destroy)
        self.btn_close["justify"] = "center"
        self.btn_close.place(x=1200, y=770, width=70, height=25)

        self.btn_open_txt = tk.Button(root, text="txt ", command=self.open_txt, state="disabled")
        self.btn_open_txt["justify"] = "center"
        self.btn_open_txt.place(x=1000, y=770, width=50, height=25)

        self.btn_start_robot = tk.Button(root, text="Start Robot", command=threading.Thread(target=self.start_rbt).start, state="disabled")
        self.btn_start_robot["justify"] = "center"
        self.btn_start_robot.place(x=850, y=770, width=100, height=25)

        btn_cls = tk.Button(root, text="Clean All", command=self.clean_all)
        btn_cls["justify"] = "center"
        btn_cls.place(x=1100, y=770, width=70, height=25)

    def clean_all(self):
        cap.release()
        cv.destroyAllWindows()

        self.lbl_start_img_t1.config(text="Tile 1")
        self.lbl_start_img_t2.config(text="Tile 2")
        self.lbl_start_rob_t1.config(text="Tile 1")
        self.lbl_start_rob_t2.config(text="Tile 2")

        self.lbl_end_img_t1.config(text="Tile 1")
        self.lbl_end_img_t2.config(text="Tile 2")
        self.lbl_end_rob_t1.config(text="Tile 1")
        self.lbl_end_rob_t2.config(text="Tile 2")

        self.place_sample(image_preview)
        self.place_sample_2(image_preview)

    def place_sample(self, sample_image):
        sample_image = sample_image.resize((540, 360), Image.ANTIALIAS)
        self.img_sample = ImageTk.PhotoImage(sample_image)
        self.l_sample.configure(image=self.img_sample)
        self.l_sample.image = sample_image

    def place_sample_2(self, sample_image_2):
        sample_image_2 = sample_image_2.resize((540, 360), Image.ANTIALIAS)
        self.img_bbox = ImageTk.PhotoImage(sample_image_2)
        self.l_bbox.configure(image=self.img_bbox)
        self.l_bbox.image = sample_image_2

    def getInput_clicked(self):
        utils.initializeTrackbars()
        entry_check = self.entry_path.get()
        if len(entry_check) == 0:
            print("Live-Cam loaded")
            self.show_frames()
        else:
            print("Image loaded")
            image = Image.open(self.entry_path.get())
            self.place_sample(image)
        self.btn_getWrap["state"] = "active"


    def show_frames(self):
        success, frame = cap.read()
        if frame is None:
            frame = numpy.array(image_preview)
        image = cv.resize(frame, (540, 360))
        # cv.imshow("grosses fenster", frame)
        cv2image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        self.place_sample(img)
        self.after_1 = self.l_sample.after(50, self.show_frames)

    def getWrap_clicked(self, trackbar_activated=False):
        entry_check = self.entry_path.get()
        if len(entry_check) == 0:
            success, frame = cap.read()
        else:
            frame = cv.imread(self.entry_path.get())

        if frame is not None:
            input_image = cv.resize(frame, (540, 360))
        else:
            input_image = cv.resize(np.asarray(image_preview), (540, 360))


        ##########################################################

        orig_image = input_image.copy()
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  # convert the image to gray scale
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Add Gaussian blur
        if not trackbar_activated:
            thres = utils.valTrackbars()
            trackbar_activated = True
        edged = cv2.Canny(blur, thres[0], thres[1])

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        doc_cnts = np.array([])

        ######################################################### changed the area part
        for contour in contours:
            # we approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
            area = cv2.contourArea(approx)
            # if we found a countour with 4 points we break the for loop
            # (we can assume that we have found our document)
            if len(approx) == 4 and area >= 8000:
                doc_cnts = approx
                break

        if len(doc_cnts) != 0:
            # apply warp perspective to get the top-down view
            warped = four_point_transform(orig_image, doc_cnts.reshape(4, 2))
            self.scanned = cv2.resize(warped, (540, 360))
            # Show the image and all the contours
            # We draw the contours on the original image not the modified one
            # Show the image and the edges
            time.sleep(0.2)

        # wenn roboter verfährt -> keine aktualisierung
        # nur wenn detect und solve gedrückt wird unteres Bild statisch
        #elif break_tranfomation == True:
        #    return


        # cv2.imshow('Original image:', input_image)
        # cv2.imshow('Edged:', edged)
        if len(doc_cnts) != 0:
            cv2.drawContours(orig_image, [doc_cnts], -1, green, 3)
            cv2.imshow("Contours of the document", orig_image)
        cv2.drawContours(input_image, contours, -1, green, 3)
        cv2.imshow("All contours", input_image)
        # cv2.imshow("Scanned", self.scanned)

        if len(doc_cnts) != 0:
            img2_colored = cv2.cvtColor(self.scanned, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(img2_colored)
            self.place_sample_2(img2)
            self.after_2 = self.l_bbox.after(50, self.getWrap_clicked)
            self.btn_getCam["state"] = "active"


    def update_label(self, text):
        self.lbl_robot_status.config(text="State of Robot: " + str(text))


    def getCam_clicked(self):
        entry_check = self.entry_path.get()
        if len(entry_check) == 0:
            success, frame = cap.read()
        else:
            frame = cv.imread(self.entry_path.get())

        success, frame = cap.read()
        input_image = cv.resize(self.scanned, (800, 600))

        ##########################################################

        zoomFactor = 1  # start zoom factor for vizualisation
        pyrfactor = 2  # factor for data reduction
        levels = 16  # number of thresholds for binarization
        autoscaling = True

        cardDetector = CardDetector(pyrfactor, levels)
        tileDetector = TileDetector(visualize=True)
        ubongoSolver = UbongoSolver()
        visualizer = Vizualizer()

        if autoscaling:
            input_image = autoscale(input_image)

        ubongoCards = cardDetector.compute(input_image.copy())
        tileContours = tileDetector.compute(input_image.copy())
        card_cnts = [card.contour for card in ubongoCards]
        ubongoTiles = tileDetector.removeCards(card_cnts, tileContours)

        self.lbl_start_img_t1.config(text="[Tile 1]:\n \nAngle: " + str(ubongoTiles[0].angle) + " deg\nX-Pos: " + str(int(ubongoTiles[0].centroid[0])) + " px\nY-Pos: " + str(int(ubongoTiles[0].centroid[1])) + " px")
        self.lbl_start_img_t2.config(text="[Tile 2]:\n \nAngle: " + str(ubongoTiles[1].angle) + " deg\nX-Pos: " + str(int(ubongoTiles[1].centroid[0])) + " px\nY-Pos: " + str(int(ubongoTiles[1].centroid[1])) + " px")

        centroid_card = [np.mean(cnt.squeeze(), axis=0) for cnt in card_cnts]
        vector = np.vectorize(np.int)
        centroid_card = vector(centroid_card)
        flatten_centroid = centroid_card.flatten()
        self.lbl_origin_card.config(text="Card Centroid: X=" + str(flatten_centroid[0]) + " Y=" + str(flatten_centroid[1]) + " in [px] in image coordinates")


        # roboter coordinates
        # koordinaten Ursprung: x = 497.2; y= 3.3 z = -129
        # bild maße: 800 x 600 px
        # maße dina4 querformat: breite:297mm höhe:210mm

        # px to mm:
        tile1_x_mm = int(497-(int(ubongoTiles[0].centroid[0])/800)*297)
        tile1_y_mm = int(3+(int(ubongoTiles[0].centroid[1])/600)*210)
        tile2_x_mm = int(497-(int(ubongoTiles[1].centroid[0])/800)*297)
        tile2_y_mm = int(3+(int(ubongoTiles[1].centroid[1])/600)*210)

        self.lbl_start_rob_t1.config(text="[Tile 1]:\n \nX-Pos: " + str(tile1_x_mm) + " mm\nY-Pos: " + str(tile1_y_mm) + " mm")
        self.lbl_start_rob_t2.config(text="[Tile 2]:\n \nX-Pos: " + str(tile2_x_mm) + " mm\nY-Pos: " + str(tile2_y_mm) + " mm")

        cardContour = input_image.copy()
        cv2.drawContours(image=cardContour, contours=card_cnts, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv.imshow("Card Contour", cardContour)

        for i, card in enumerate(ubongoCards):

            if card.playFieldPattern is None or len(card.blocks) == 0:
                continue

            card.isValid, card.blockPositions = ubongoSolver.findUbongoSolution(card.playFieldPattern, card.blocks)

        np.savetxt('tile1.txt', [int(ubongoTiles[0].angle)] + [int(ubongoTiles[0].centroid[0])] + [int(ubongoTiles[0].centroid[1])], fmt="%s")
        np.savetxt('tile2.txt', [int(ubongoTiles[1].angle)] + [int(ubongoTiles[1].centroid[0])] + [int(ubongoTiles[1].centroid[1])], fmt="%s")

        vizImage = visualizer.createVizImage(input_image.copy(), ubongoCards)
        # cv2.imshow("Solution", vizImage)
        highlighted_centroids = vizImage.copy()
        highlighted_centroids[int(ubongoTiles[0].centroid[1]):int(ubongoTiles[0].centroid[1])+5, int(ubongoTiles[0].centroid[0]):int(ubongoTiles[0].centroid[0])+5] = (0, 255, 255)
        highlighted_centroids[int(ubongoTiles[1].centroid[1]):int(ubongoTiles[1].centroid[1])+5, int(ubongoTiles[1].centroid[0]):int(ubongoTiles[1].centroid[0])+5] = (255, 255, 0)
        cv2.imshow("Solution with centroids", highlighted_centroids)

        crop_img = vizImage[0:250, 800:vizImage.shape[1]]
        cv2.imshow("Only Solution", crop_img)
        print(crop_img.shape)

        # color filter
        colorfilter = colorfiltering.ColorFilter()
        cyan_tile = colorfilter.get_cyan_tile(image=crop_img)
        yellow_tile = colorfilter.get_yellow_tile(image=crop_img)
        # cv.imshow('result cyan', cyan_tile)  # Image after bitwise operation
        # cv.imshow('result yellow', yellow_tile)  # Image after bitwise operation

        # centroids of tiles in solution
        cyan_tile_detected, cyan_centroid_x, cyan_centroid_y = colorfilter.get_cyan_centroid(image=cyan_tile)
        cv.imshow("cyan detcted", cyan_tile_detected)
        print("Cyan X Centroid: " + str(cyan_centroid_x))
        print("Cyan Y Centroid: " + str(cyan_centroid_y))

        yellow_tile_detected, yellow_centroid_x, yellow_centroid_y = colorfilter.get_yellow_centroid(image=yellow_tile)
        cv.imshow("yellow detcted", yellow_tile_detected)
        print("Yellow X Centroid: " + str(yellow_centroid_x))
        print("Yellow Y Centroid: " + str(yellow_centroid_y))

        # distance from card centroid to tile centroid in the solution card
        distance_x_cyan, distance_y_cyan = colorfilter.dist_cyan_centroid(c_c_x=cyan_centroid_x, c_c_y=cyan_centroid_y)
        distance_x_yellow, distance_y_yellow = colorfilter.dist_yellow_centroid(c_y_x=yellow_centroid_x, c_y_y=yellow_centroid_y)
        print(distance_x_cyan, distance_y_cyan)
        print(distance_x_yellow, distance_y_yellow)

        # centroid of card in image coordinate system
        centroid_card_x = flatten_centroid[0]
        centroid_card_y = flatten_centroid[1]
        print(centroid_card_x, centroid_card_y)


        # Breite Karte Bildbereich / Breite Karte Lösungsbereich (140px) = Breitenfakrot
        # Höhe Karte Bildbereich / Höhe Karte Lösungsbereich (200px) = Höhenfaktor
        # 600 px = 210mm --> 1 mm = 2.857 px/mm
        # 800 px = 297mm --> 1 mm = 2.694 px/mm

        width_factor = (36*2.857)/140
        height_factor = (60*2.694)/200


        # distance_x * Breitenverhältnis = Abstand X im Bildbereich vom Teil zum Schwerpunkt der Karte
        # distance_y * Höhenverhältnis = Abstand Y im Bildbereich vom Teil zum Schwerpunkt der Karte
        # distances in image coordinates and px

        x_offset_cyan = int(centroid_card_x + (distance_x_cyan*width_factor))
        y_offset_cyan = int(centroid_card_y + (distance_y_cyan*height_factor))
        x_offset_yellow = int(centroid_card_x + (distance_x_yellow * width_factor))
        y_offset_yellow = int(centroid_card_y + (distance_y_yellow * height_factor))

        print(x_offset_cyan)
        print(y_offset_cyan)
        print(x_offset_yellow)
        print(y_offset_yellow)

        highlighted_end_pos = vizImage.copy()
        highlighted_end_pos[y_offset_cyan:y_offset_cyan + 5, x_offset_cyan:x_offset_cyan + 5] = (255, 255, 0)
        highlighted_end_pos[y_offset_yellow:y_offset_yellow + 5, x_offset_yellow:x_offset_yellow + 5] = (0, 255, 255)
        cv2.imshow("End Pos of Tiles", highlighted_end_pos)


        # darstellung in Roboterkoordinaten --> Ablageposition
        # koordinaten Ursprung: x = 497.2; y= 3.3 z = -129
        # bild maße: 800 x 600 px
        # maße dina4 querformat: breite:297mm höhe:210mm

        # px to mm:
        x_offset_cyan_mm = int(497 - ((x_offset_cyan / 800) * 297))
        y_offset_cyan_mm = int(3 + ((y_offset_cyan / 600) * 210))
        x_offset_yellow_mm = int(497 - ((x_offset_yellow / 800) * 297))
        y_offset_yellow_mm = int(3 + ((y_offset_yellow / 600) * 210))

        print(x_offset_cyan_mm)
        print(y_offset_cyan_mm)
        print(x_offset_yellow_mm)
        print(y_offset_yellow_mm)

        self.lbl_end_img_t1.config(text="[Tile 1]:\n \nX-Pos: " + str(int(x_offset_cyan)) + " px\nY-Pos: " + str(int(y_offset_cyan)) + " px")
        self.lbl_end_img_t2.config(text="[Tile 2]:\n \nX-Pos: " + str(int(x_offset_yellow)) + " px\nY-Pos: " + str(int(y_offset_yellow)) + " px")

        self.lbl_end_rob_t1.config(text="[Tile 1]:\n \nX-Pos: " + str(int(x_offset_cyan_mm)) + " mm\nY-Pos: " + str(int(y_offset_cyan_mm)) + " mm")
        self.lbl_end_rob_t2.config(text="[Tile 2]:\n \nX-Pos: " + str(int(x_offset_yellow_mm)) + " mm\nY-Pos: " + str(int(y_offset_yellow_mm)) + " mm")

        global t1_pickup_x, t1_pickup_y, t1_endpos_x, t1_endpos_y, t2_pickup_x, t2_pickup_y, t2_endpos_x, t2_endpos_y
        t1_pickup_x = tile1_x_mm
        t1_pickup_y = tile1_y_mm
        t2_pickup_x = tile2_x_mm
        t2_pickup_y = tile2_y_mm
        t1_endpos_x = int(x_offset_cyan_mm)
        t1_endpos_y = int(y_offset_cyan_mm)
        t2_endpos_x = int(x_offset_yellow_mm)
        t2_endpos_y = int(y_offset_yellow_mm)

        img2 = Image.fromarray(input_image)
        self.place_sample_2(img2)

        self.btn_open_txt["state"] = "active"
        self.btn_start_robot["state"] = "active"

    def open_txt(self):
        os.startfile('tile1.txt')
        os.startfile('tile2.txt')

    def start_rbt(self):
        print("start Robot")

        self.lbl_robot_status["bg"] = "red"
        self.update_label("Caution! Robot moving. ")

        precise.connect()
        x, y, z = precise.wherexyz()
        print("current x pos:", x)
        print("current y pos:", y)
        print("current z pos:", z)

        """Tile 1 pickup-routine"""
        # startpos -> tile1 -> z runter -> pin an -> z hoch -> verfahren endpos -> z runter -> pin aus -> z hoch
        precise.movexyz()  # Home
        precise.movexyz(t1_pickup_x, t1_pickup_y, globals.home_z)
        self.update_label("Caution! Robot picking up tile 1. ")
        precise.movexyz(t1_pickup_x, t1_pickup_y, globals.down_z)  # z down
        precise.enableDIO(33)
        precise.movexyz(t1_pickup_x, t1_pickup_y, globals.home_z)  # z up
        precise.movexyz(t1_endpos_x, t1_endpos_y, globals.home_z)
        self.update_label("Caution! Robot putting down tile 1. ")
        precise.movexyz(t1_endpos_x, t1_endpos_y, globals.down_z)  # z down
        precise.disableDIO(33)
        precise.movexyz(t1_endpos_x, t1_endpos_y, globals.home_z)  # z up
        self.update_label("Caution! Robot moving. ")

         # Tile 2 pickup-routine

        # startpos -> tile2 -> z runter -> pin an -> z hoch -> verfahren endpos -> z runter -> pin aus -> z hoch
        precise.movexyz(t2_pickup_x, t2_pickup_y, globals.home_z)
        self.update_label("Caution! Robot picking up tile 1. ")
        precise.movexyz(t2_pickup_x, t2_pickup_y, globals.down_z)  # z down
        precise.enableDIO(33)
        precise.movexyz(t2_pickup_x, t2_pickup_y, globals.home_z)  # z up
        precise.movexyz(t2_endpos_x, t2_endpos_y, globals.home_z)
        self.update_label("Caution! Robot putting down tile 1. ")
        precise.movexyz(t2_endpos_x, t2_endpos_y, globals.down_z)  # z down
        precise.disableDIO(33)
        precise.movexyz(t2_endpos_x, t2_endpos_y, globals.home_z)  # z up
        self.update_label("Caution! Robot moving. ")
        precise.movexyz()  # Home

        precise.disconnect()

        self.lbl_robot_status["bg"] = "green"
        self.update_label("Robot has finished its movement. ")


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()
