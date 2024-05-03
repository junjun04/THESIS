from tkinter import *
from tkinter import filedialog
from datetime import date
from PIL import Image, ImageTk
from tkinter import messagebox
import argparse
import numpy as np
import os
import cv2
import colorsys
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
import pandas as pd
from scipy.stats import t


root = Tk()
root.title("TILAPIA FISH FRESHNESS ASSESSMENT SYSTEM")

width = 1000  # Width
height = 600  # Height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width/2) - (width/2)
y = (screen_height/2) - (height/2)
root.geometry('%dx%d+%d+%d' % (width, height, x, y))
root.configure(bg='gray')


# PP-YOLO DATASET APPLICATION

def selected():
    global img_path, img
    img_path = filedialog.askopenfilename(initialdir=os.getcwd())

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image", default=img_path,
                        help="image for prediction")
    parser.add_argument(
        "--config", default='data/ppyolo_tilapia.cfg')
    parser.add_argument(
        "--weights", default='data/ppyolo_tilapia.weights')
    parser.add_argument(
        "--names", default='data/obj.names',
    )
    args = parser.parse_args()

    color_freshness.config(text="Color Freshness Analysis:",
                           font='Helvetica 16 bold ', fg='white', bg='gray')
    label_freshness.config(text="Freshness Analysis:",
                           font='Helvetica 16 bold ', fg='white', bg='gray')
    average_hsv.config(text="Average Color of HSV:",
                       font='Helvetica 16 bold ', fg='white', bg='gray')
    average_rgb.config(text="Average Color of RGB:",
                       font='Helvetica 16 bold ', foreground='white', bg='gray')
    CONF_THRESH, NMS_THRESH = 0.5, 0.5

# DARKNET
    # Load the network
    net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Get the output layer from PPYOLO
    layers = net.getLayerNames()
    layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

    # BOUNDING BOX
    img = cv2.imread(args.image)
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(layers)

    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (
                    detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
    indices = cv2.dnn.NMSBoxes(
        b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

    # Draw the filtered bounding boxes with their class to the image
    with open(args.names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for index in indices:
        x, y, w, h = b_boxes[index]
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[index], 2)
        cv2.putText(img, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[index],
                    2)

    # Create and write the bounding box coordinates to a .txt file
    txt = open("outputs/coordinates.txt", "w")
    coordinates = " ".join(str(i) for i in b_boxes[0])
    txt.write(coordinates)
    # Show the selected image
    cv2.imwrite("outputs/prediction.jpg", img)
    img = Image.open("outputs/prediction.jpg")
    img.thumbnail((410, 410))
    fish_image = ImageTk.PhotoImage(img)
    canvas2.create_image(480, 210, image=fish_image)
    canvas2.image = fish_image


def watershed():

    global freshness
    global display_btn
    global btn_sop
    try:
        image = cv2.imread(img_path)
        messagebox.showinfo(
            title="THE SYSTEM IS RUNNING . . . ", message="Please wait for a moment, performing watershed algorithm and HSV Channels for Extraction.\nPress OK to continue")

        with open('outputs/coordinates.txt', 'r') as file:
            fp = file.read()
        coord = fp.split()
        rectangle = (int(coord[0]), int(coord[1]),
                     int(coord[2]), int(coord[3]))

        # Watershed Algorithm  Extraction

        cropped = image[rectangle[1]:rectangle[1] +
                        rectangle[3], rectangle[0]:rectangle[0] + rectangle[2]]
        # Convert the segmented image to grayscale
        grayscale = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # Thresholding to create a binary mask for the markers
        _, thresh = cv2.threshold(
            grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Perform morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, kernel, iterations=4)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=4)
        # Finding sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist_transform, 0.2 * dist_transform.max(), 255, 0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        # Apply Watershed algorithm
        markers = cv2.watershed(cropped, markers)
        # Mark watershed boundaries in yellow (you can change the color)
        cropped[markers == -1] = [0, 255, 255]

        cv2.imwrite("outputs/watershed.png", cropped)
        size = cropped.shape
        print(size[0], size[1])

        cv2.imwrite("outputs/mask.png", thresh)

        b, g, r = cv2.split(cropped)
        alpha = np.zeros_like(thresh)
        rgba = cv2.merge((b, g, r, alpha))
        rgba[:, :, 3] = thresh
        cv2.imwrite("outputs/segmented.png", rgba)

        clrd = cv2.imread("outputs\segmented.png")
        mskd = cv2.imread("outputs\mask.png")

        # Specify the rgb color of the given mask by calculating the distance point
        py = clrd.shape[0]
        px = clrd.shape[1]
        count = 0
        ar = 0
        ag = 0
        ab = 0

        for y in range(0, py):
            for x in range(0, px):
                if mskd[y, x, 2] == 255 and mskd[y, x, 1] == 255 and mskd[y, x, 0] == 255:
                    count = count+1
                    color_red = clrd[y, x, 2]
                    color_green = clrd[y, x, 1]
                    color_blue = clrd[y, x, 0]
                    ar = ar+color_red
                    ag = ag+color_green
                    ab = ab+color_blue

        # Average RGB Values
        avgRed = round(ar/count)
        avgGreen = round(ag/count)
        avgBlue = round(ab/count)

        # Store the Average RGB Values to convert into Average HSV Values
        redGreenBlue = [avgRed, avgGreen, avgBlue]

        red1, green1, blue1 = [x / 255.0 for x in redGreenBlue]

        # Conversion of RGB to HSV
        hue1, saturation1, value1 = colorsys.rgb_to_hsv(red1, green1, blue1)

        # Scale values to typical HSV ranges
        hue1 *= 360
        saturation1 *= 100
        value1 *= 100

        hsv_values = [hue1, saturation1, value1]

        # Round each HSV value
        rounded_h = round(hsv_values[0])
        rounded_s = round(hsv_values[1])
        rounded_v = round(hsv_values[2])


# CONDITIONAL STATEMENT

        if 11 < rounded_h <= 16:

            freshness = "NOT FRESH"
            color_freshness1 = "REDDISH BROWN"
        else:
            freshness = "OLD"
            color_freshness1 = "BROWN"
        if rounded_h <= 11:

            freshness = "FRESH"
            color_freshness1 = "DARK/BRIGHT RED"
        else:
            freshness = "NOT FRESH"
            color_freshness1 = "REDDISH BROWN"

        if rounded_h >= 17:
            freshness = "OLD"
            color_freshness1 = "BROWN"

        # Get image name
        img_name = img_path.split('/')[-1]

        # Get current date
        today = date.today()

        watershed_image = Image.open("outputs/segmented.png")
        # img.thumbnail((400, 400))
        watershed_image = watershed_image.resize((410, 410), Image.LANCZOS)
        mask_image = ImageTk.PhotoImage(watershed_image)
        canvas2.create_image(480, 210, image=mask_image)
        canvas2.image = mask_image

        color_freshness_analysis = "Color Freshness Analysis: {}".format(
            color_freshness1)

        freshness_analysis = "Freshness Analysis: {}".format(freshness)

        rgb_channels = "Average Value of RGB: ({}, {}, {})".format(
            avgRed, avgGreen, avgBlue)

        hsv_channels = "Average Value of HSV: ({}, {}, {})".format(
            rounded_h, rounded_s, rounded_v)

        display_btn = Button(root, text="DISPLAY", font='Aerial 13 ', width=15,
                             relief=GROOVE, command=setup_gui, cursor='hand2', activebackground='light blue', bg='SystemButtonFace')
        display_btn.place(x=650, y=540)

        btn_sop = Button(root, text="S.O.P", width=15,
                         font='ariel 13 ', relief=GROOVE,  command=sop, cursor='hand2', activebackground='light blue', bg='SystemButtonFace')
        btn_sop.place(x=820, y=540)

        display_btn.bind("<Enter>", on_enter)
        display_btn.bind("<Leave>", on_leave)

        btn_sop.bind("<Enter>", on_enter_sop)
        btn_sop.bind("<Leave>", on_leave_sop)

        color_freshness.config(text=color_freshness_analysis,
                               font='Helvetica 16 bold ', fg='white', bg='gray')

        label_freshness.config(text=freshness_analysis,
                               font='Helvetica 16 bold ', fg='white', bg='gray')

        average_hsv.config(text=hsv_channels,
                           font='Helvetica 16 bold ', fg='white', bg='gray')

        average_rgb.config(text=rgb_channels,
                           font='Helvetica 16 bold ', fg='white', bg='gray')

        success_box = "{}\n{}\n{}\n{}\n".format(
            freshness_analysis, rgb_channels, hsv_channels,  color_freshness_analysis)

        messagebox.showinfo(title="SUCCESS", message=success_box)

        file = open("outputs/logs.txt", "a")
        log = "{}\t{}\t\t{}\n".format(today, img_name, freshness)
        file.write(log)
        file.close()

    except:
        messagebox.showwarning(
            title="WATERSHED", message="Image path not defiend")


fish_image = None
display_btn = None
btn_predicted = None
btn_mask = None
btn_segmented = None
btn_watershed = None
btn_sop = None


def on_enter(event):
    global display_btn
    if display_btn:
        display_btn.config(bg='dark blue', fg='white')


def on_leave(event):
    global display_btn
    if display_btn:
        display_btn.config(bg='SystemButtonFace', fg='black')


def on_enter_sop(event):
    global btn_sop
    if btn_sop:
        btn_sop.config(bg='dark blue', fg='white')


def on_leave_sop(event):
    global btn_sop
    if btn_sop:
        btn_sop.config(bg='SystemButtonFace', fg='black')


def display_image(canvaS, image_path):
    image1 = Image.open(image_path)
    image1 = image1.resize((410, 410), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image1)
    canvaS.create_image(480, 210, image=photo)
    canvaS.image = photo


# def open_image_file(canvas):
#     file_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File",
#                                            filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.gif"), ("All files", "*.*")))
#     if file_path:
#         display_image(canvas, file_path)


def setup_gui():
    global btn_watershed, btn_predicted, btn_mask, btn_segmented
    top = Toplevel(root)
    top.title("TILAPIA FISH FRESHNESS ASSESSMENT SYSTEM")
    width = 1000
    height = 500
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    top.geometry('%dx%d+%d+%d' % (width, height, x, y))
    root.configure(bg='gray')
    canvas = Canvas(top, width=960, height=360, relief=RIDGE, bd=2)
    canvas.place(x=15, y=10)
    canvas.configure(bg='gray')
    folder_path = os.path.dirname(os.path.abspath(__file__))

    btn_predicted = Button(top, text="Predicted Image", cursor='hand2', activebackground='light blue', bg='SystemButtonFace', width=15, font='ariel 13',
                           relief=GROOVE, command=lambda: display_image(canvas, os.path.join(folder_path, 'outputs/prediction.jpg')))
    btn_predicted.place(x=25, y=440)

    btn_watershed = Button(top, text="Watershed Image", cursor='hand2', activebackground='light blue', bg='SystemButtonFace', width=15, font='ariel 13',
                           relief=GROOVE, command=lambda: display_image(canvas, os.path.join(folder_path, 'outputs/watershed.png')))
    btn_watershed.place(x=200, y=440)

    btn_segmented = Button(top, text="Segmented Image", cursor='hand2', activebackground='light blue', bg='SystemButtonFace', width=15, font='ariel 13',
                           relief=GROOVE, command=lambda: display_image(canvas, os.path.join(folder_path, 'outputs/segmented.png')))
    btn_segmented.place(x=550, y=440)

    btn_mask = Button(top, text="Mask Image", cursor='hand2', activebackground='light blue', bg='SystemButtonFace', width=15, font='ariel 13',
                      relief=GROOVE, command=lambda: display_image(canvas, os.path.join(folder_path, 'outputs/mask.png')))
    btn_mask.place(x=375, y=440)

    top.configure(bg='gray')
    root.mainloop()


def legend_gui():
    top2 = Toplevel(root)
    top2.title("TILAPIA FISH FRESHNESS ASSESSMENT SYSTEM")
    width = 1000
    height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    top2.geometry('%dx%d+%d+%d' % (width, height, x, y))
    top2.configure(bg='gray')

    # Add a label for text display
    label_legend = Label(top2, text="TILAPIA FISH FRESHNESS ASSESSMENT SYSTEM", font=(
        'Arial', 20, 'bold'), bg='gray', fg='white')
    label_legend.pack(anchor='center', pady=10)

    # Add a paragraph of text
    text = Text(top2, wrap=WORD, font=('Arial', 12), bg='gray',
                fg='white', height=20, cursor='hand2')
    text.pack(expand=True, fill='both', padx=20, pady=10)
    # text.insert(
    #     END, "This is a panel explaining the Tilapia Fish Freshness Assessment System and its functionalities.\n\n")
    text.insert(END, "\nTilapia Fish Freshness Assessment System is a windows system/software that can assess or tell the freshness of a tilapia fish using gill's color. The freshness analysis are labeled into FRESH, NOT FRESH and OLD. \n\n", ('legend1',))
    text.insert(END, "FRESHNESS ANALYSIS: \n\n", ('legend2',))
    text.insert(END, "FRESH:-> Dark Red or Bright Red in terms of gill color\n\nNOT FRESH: -> Reddish Brown or Pink in terms of gill color \n\nOLD:-> Brown or Gray in terms of gill color \n\n", ('legend1'))
    text.insert(END, "FUNCTIONALITIES: \n\n", ('legend2',))
    text.insert(END, "SELECT IMAGE:-> This button is for user to be able select an image of a tilapia fish from a local directory \n\nTEST: -> This button is for the user to be able to test the system and apply the algorithms such as PP-YOLO Algorithm, Watershed Algorithm and HSV Channels\n\n After testing, there will be a button that will pop-up named DISPLAY\n\nDISPLAY: -> This button is for the user if they want to see the output images during the testing phase ", ('legend1'))

    # Configure tags for different parts of the text
    text.tag_config('legend1', font=(
        'Arial', 12, 'normal'), foreground='white')
    text.tag_config('legend2', font=(
        'Arial', 12, 'bold'), foreground='white')
    # text.tag_config('legend3', font=('Arial', 12, 'normal'), foreground='white')

    text.configure(state='disabled')

    root.mainloop()

# GUI OF RESEARCH QUESTIONS


def sop():
    data12 = {
        'Proposed System': ['Fresh', 'Not Fresh', 'Old'],
        'Total No. of Test Cases': [39, 29, 15],
        'Total No. Correctly Assess': [38, 18, 15],
        'Accuracy Rate ': ['97.43%', '58.62%', '100%']
    }

    df = pd.DataFrame(data12)

    # Calculate total matches and accuracy
    total_matches = df['Total No. Correctly Assess'].sum()
    total_cases = df['Total No. of Test Cases'].sum()
    accuracy = round((total_matches / total_cases) * 100, 2)

    total_row = pd.DataFrame({
        'Proposed System': ['Total Number'],
        'Total No. of Test Cases': [f'{total_cases}'],
        'Total No. Correctly Assess': [f'{total_matches} Correct Matches'],
        'Accuracy Rate ': [''],
    })

    accuracy_row = pd.DataFrame({
        'Proposed System': ['Accuracy of the System = '],
        'Total No. of Test Cases': ['(71 divded by '],
        'Total No. Correctly Assess': [' 83 ) x 100'],
        'Accuracy Rate ': [f'{accuracy}%'],
    })

    # Concatenate the rows to the DataFrame
    df = pd.concat([df, total_row, accuracy_row], ignore_index=True)

    root = tk.Tk()
    root.title("RESEARCH QUESTIONS OF THE STUDY")
    width = 1000  # Width
    height = 600  # Height
    screen_width = root.winfo_screenwidth()  # Width of the screen
    screen_height = root.winfo_screenheight()  # Height of the screen
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))
    root.configure(bg='white')

    description_text = (
        "1.) What is the accuracy of the proposed system (Tilapia fish freshness assessment "
        "by gill color using PP-YOLO algorithm and Watershed algorithm with the usage of "
        "HSV color channels for feature extraction)?"
    )
    description_label = ttk.Label(
        root, text=description_text, wraplength=960, justify='left', font=('Helvetica', 12), background='white')
    description_label.pack(padx=10, pady=10)

    frame1 = ttk.Frame(root)
    frame1.pack()

    treeview1 = ttk.Treeview(frame1, columns=list(df.columns), show='headings')
    treeview1.pack()

    style = ttk.Style()
    style.configure('Treeview.Heading', font=('Helvetica', 14))
    style.configure("Treeview", rowheight=40)  # Adjust row height

    for col in df.columns:
        treeview1.heading(col, text=col)

    for _, row in df.iterrows():
        treeview1.insert("", "end", values=list(row))
    header_widths = {col: len(col) for col in df.columns}
    data_widths = {col: df[col].astype(str).map(
        len).max() for col in df.columns}
    for col in df.columns:
        col_width = max(header_widths[col], data_widths[col]) * 10
        treeview1.column(col, width=col_width, anchor='center')
    for col in df.columns:
        treeview1.tag_configure(col, anchor='center')

    # Show grid lines
    style.map("Treeview", background=[
        ('selected', '#347083')], foreground=[('selected', 'white')])
    style.configure("Treeview", rowheight=20, font=('Helvetica', 12),
                    fieldbackground="black",
                    background="white", relief="flat")
    style.configure("Treeview.Heading", font=(
        'Helvetica', 12), relief="flat", bd=0)
    treeview1.configure(style="Treeview")

    description_text2 = (
        "2.) Is there a significant difference between the accuracy of the proposed system "
        "(Tilapia fish freshness assessment by gill color using PP-YOLO algorithm and "
        "Watershed algorithm with the usage of HSV color channels for feature extraction), "
        "and the previous system of Cortez et. al (2022) (Tilapia fish freshness evaluation "
        "by gill color using YOLOv3 and Grabcut algorithm and the utilization of RGB channels "
        "for feature extraction) in determining the freshness of a tilapia fish?"
    )
    description_label2 = ttk.Label(
        root, text=description_text2, wraplength=960, justify='left', font=('Helvetica', 12), background='white')
    description_label2.pack(padx=10, pady=10)

    # PAIRED T TEST

    frame3 = ttk.Frame(root)
    frame3.pack()

    data3 = {
        'Proposed System': ["System Accuracy"],
        'Mean': [3],
        'T-statistics': [4.242641],
        'P-value': ['0.075'],
        'Decision': ["Reject Null Hypothesis"],
        'Remarks': ["There is a significant difference"]
    }
    # df = pd.DataFrame(data3)

    treeview2 = ttk.Treeview(
        frame3, columns=list(data3.keys()), show='headings')
    treeview2.pack()

    style = ttk.Style()
    style.configure('Treeview.Heading', font=('Helvetica', 16))
    style.configure("Treeview", rowheight=5)  # Adjust row height

    for col in data3.keys():
        treeview2.heading(col, text=col)

    treeview2.insert("", "end", values=list(data3.values()))

    # Calculate column widths for Table 2
    header_widths2 = {col: len(col) for col in data3.keys()}
    data_widths2 = {col: len(str(data3[col][0])) for col in data3.keys()}
    for col in data3.keys():
        col_width = max(header_widths2[col], data_widths2[col]) * 9
        treeview2.column(col, width=col_width, anchor='center')
    for col in data3.keys():
        treeview2.tag_configure(col, anchor='center')

    # Show grid lines
    style.map("Treeview", background=[
        ('selected', '#347083')], foreground=[('selected', 'white')])
    style.configure("Treeview", rowheight=5, font=('Helvetica', 16),
                    fieldbackground="black",
                    background="white", relief="flat")
    style.configure("Treeview.Heading", font=(
        'Helvetica', 12), relief="flat", bd=0)
    treeview2.configure(style="Treeview")

    root.mainloop()

# Your second data table
    data3 = {
        'System': ['System Accuracies'],
        'Mean': [3],
        'T-statistics': [4.242641],
        'P-value': ['0.075'],
        'Decision': ['Reject Null Hypothesis'],
        'Remarks': ['There is a significant difference']
    }

    root.mainloop()


def on_enter_Select(event):
    btn_Select.config(bg='dark blue', fg='white')


def on_leave_Select(event):
    btn_Select.config(bg='SystemButtonFace', fg='black')


def on_enter_Legend(event):
    btn_Legend.config(bg='dark blue', fg='white')


def on_leave_Legend(event):
    btn_Legend.config(bg='SystemButtonFace', fg='black')


def on_enter_Test(event):
    btn_Test.config(bg='dark blue', fg='white')


def on_leave_Test(event):
    btn_Test.config(bg='SystemButtonFace', fg='black')


# create canvas to display image
canvas2 = Canvas(root, width="960", height="360",
                 bd=2, relief=RIDGE, bg='gray')
canvas2.place(x=15, y=10)
# canvas2.config(bg='white')

# freshness analyis label
label_freshness = Label(root, text="Freshness Analysis:",
                        font='Helvetica 16 bold', fg='white', bg='gray')
label_freshness.place(x=20, y=400)

color_freshness = Label(root, text="Color Freshness Analysis:",
                        font='Helvetica 16 bold', fg='white', bg='gray')
color_freshness.place(x=20, y=470)

# average RGB channels label
average_rgb = Label(root, text="Average Color of RGB:",
                    font='Helvetica 16 bold ', fg='white', bg='gray')
average_rgb.place(x=610, y=400)

# average HSV channels label
average_hsv = Label(root, text="Average Color of HSV:",
                    font='Helvetica 16 bold ', fg='white', bg='gray')
average_hsv.place(x=610, y=470)


# create buttons
btn_Select = Button(root, text="SELECT IMAGE", width=15,
                    font='ariel 13 ', relief=GROOVE,  command=selected, cursor='hand2', activebackground='light blue', bg='SystemButtonFace')
btn_Select.place(x=25, y=540)
btn_Select.bind("<Enter>", on_enter_Select)
btn_Select.bind("<Leave>", on_leave_Select)

btn_Test = Button(root, text="TEST", width=15, font='ariel 13 ',
                  relief=GROOVE,  command=watershed, cursor='hand2', activebackground='light blue', bg='SystemButtonFace')
btn_Test.place(x=220, y=540)
btn_Test.bind("<Enter>", on_enter_Test)
btn_Test.bind("<Leave>", on_leave_Test)

btn_Legend = Button(root, text="LEGEND", width=15, font='ariel 13 ',
                    relief=GROOVE, command=legend_gui, cursor='hand2', activebackground='light blue', bg='SystemButtonFace')
btn_Legend.place(x=410, y=540)
btn_Legend.bind("<Enter>", on_enter_Legend)
btn_Legend.bind("<Leave>", on_leave_Legend)
# open_image_file(canvaS)
root.mainloop()
