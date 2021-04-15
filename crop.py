#!/usr/bin/env python
###################### INFORMATION ##############################
#          Extract the table of content (TOC) from Reichanzeiger
#Program:  **crop**
#Info:     **Python3**
#Author:   **Jan Kamlah**
#Date:     **08.04.2021**
####################### IMPORT ##################################
import argparse
import copy
import os
import warnings

import numpy as np
import scipy.misc as misc
import skimage as ski
import skimage.color as color
import skimage.filters.thresholding as th
import skimage.morphology as morph
import skimage.transform as transform
from scipy.ndimage import measurements
from skimage.io import imread, imsave


####################### CLASSES & METHODS ###########################
class Clippingmask():
    def __init__(self, image):
        self.height_start, self.width_start = 0, 0
        if len(image.shape) > 2:
            self.height_stop, self.width_stop, self.rgb = image.shape
        else:
            self.height_stop, self.width_stop = image.shape
        self.user = None

class ImageParam():
    def __init__(self, image, input):
        if len(image.shape) > 2:
            self.height, self.width, self.rgb = image.shape
        else:
            self.height, self.width = image.shape
        self.path = os.path.dirname(input)
        self.pathout = os.path.normpath(os.path.dirname(input)+"/TOC-Extraction/")
        self.deskewpath = None
        self.name = os.path.splitext(os.path.basename(input))[0]

class Linecoords():
    def __init__(self, binary, value ,object):
        self.height_start = object[0].start
        self.height_stop = object[0].stop
        self.width_start = object[1].start
        self.width_stop = object[1].stop
        self.middle = None
        self.object = object
        self.object_value = value
        self.object_matrix = copy.deepcopy(binary[object])
        self.segmenttype = None

class SpliceParam():
    def __init__(self, input, parts):
        self.name = os.path.splitext(input)[0]
        self.segment = parts[len(parts)-2]
        self.segmenttype = parts[len(parts)-1]

####################### FUNCTIONS ##################################
def create_dir(newdir):
    if not os.path.isdir(newdir):
        try:
            os.makedirs(newdir)
            print(newdir)
        except IOError:
            print(("cannot create %s directoy" % newdir))

def crop_lcol(args, image, image_param, list_linecoords, clippingmask):
    # Find left column
    pixelheight = set_pixelground(image_param.height)
    image = np.rot90(image, args.horlinepos)
    for idx, linecoords in enumerate(list_linecoords):
        # Header
        if idx == 0:
            if not args.quiet: print("header")
            roi = image[0:linecoords.height_start - 2, 0:image_param.width]  # region of interest
            roi = np.rot90(roi, 4 - args.horlinepos)
            with warnings.catch_warnings():
                # Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
        # Crop middle segments
        if linecoords.segmenttype == 'B':
            if not args.quiet: print("blank")
            # Add sum extra space to the cords
            roi = image[linecoords.height_start + 2 - pixelheight(
                args.addstartheightc):linecoords.height_stop - 2 + pixelheight(args.addstopheightc),
                  linecoords.width_start:linecoords.width_stop]  # region of interest
            roi = np.rot90(roi, 4 - args.horlinepos)
            with warnings.catch_warnings():
                # Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                if args.horlinetype == 1:
                    idx = len(list_linecoords) - idx
                if 'c' in args.croptypes:
                    pass
        if linecoords.segmenttype == 'L':
            # Fixing column size
            if idx == 0:
                print("line-first")
                # linecoords.height_start = clippingmask.height_start + 17
            if not args.quiet: print("line")
            roi = image[
                  linecoords.height_start - pixelheight(args.addstartheightab):linecoords.height_stop + pixelheight(
                      args.addstopheightab),
                  0:linecoords.width_stop - 2]  # region of interest
            roi = np.rot90(roi, 4 - args.horlinepos)
            with warnings.catch_warnings():
                # Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                if args.horlinetype == 1 and 'b' in args.croptypes:
                    idx = len(list_linecoords) - idx
                elif 'a' in args.croptypes:
                    return roi
            roi = image[
                  linecoords.height_start - pixelheight(args.addstartheightab):linecoords.height_stop + pixelheight(
                      args.addstopheightab),
                  0 + 1:clippingmask.width_stop]
            roi = np.rot90(roi, 4 - args.horlinepos)
            with warnings.catch_warnings():
                # Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                if args.horlinetype == 1 and 'a' in args.croptypes:
                    return roi
                elif 'a' in args.croptypes:
                    return roi
    return None

def cropping_lcol(imgpath, args):
    # Main cropping function that deskew, analyse and crops the image
    # read image
    print(f"Find toc in {imgpath}")
    try:
        image = imread("%s" % imgpath)
        image_param = ImageParam(image, imgpath)
        if args.imgmask != [0.0, 1.0, 0.0, 1.0]:
            image = image[int(args.imgmask[0]*image_param.height):int(args.imgmask[1]*image_param.height),
                    int(args.imgmask[2]*image_param.width):int(args.imgmask[3]*image_param.width)]
            image_param = ImageParam(image, imgpath)
    except IOError:
        print(("cannot open %s" % imgpath))
        return 1
    create_dir(image_param.pathout)
    ####################### ANALYSE - LINECOORDS #######################
    print("start linecoord-analyse")
    clippingmask = Clippingmask(image)
    border, labels, list_linecoords, topline_width_stop = linecoords_analyse(args, image, image_param, clippingmask)
    ####################### CROP #######################################
    print("start crop lcol")
    lcol = crop_lcol(args, image, image_param, list_linecoords, clippingmask)
    return lcol

def cropping_toc(lcol, args):
    image_param = ImageParam(lcol, args.input)
    if args.imgmask != [0.0, 1.0, 0.0, 1.0]:
        lcol = lcol[int(args.imgmask[0] * image_param.height):int(args.imgmask[1] * image_param.height),
                int(args.imgmask[2] * image_param.width):int(args.imgmask[3] * image_param.width)]
        image_param = ImageParam(lcol, args.input)
    clippingmask = Clippingmask(lcol)
    border, labels, list_linecoords, topline_width_stop = linecoords_analyse(args, lcol, image_param, clippingmask, get_toc=True)
    ####################### CROP #######################################
    print("start crop toc")
    tocpath = crop_toc(args, lcol, image_param, list_linecoords)
    return tocpath

def crop_toc(args, image, image_param, list_linecoords):
    # Find left column
    create_dir(image_param.pathout+os.path.normcase("/"+image_param.name.split(".",1)[0]+"/"))
    filepath = image_param.pathout+os.path.normcase("/"+image_param.name.split(".",1)[0]+"/")+image_param.name
    image = np.rot90(image, args.horlinepos)
    imsave("%s_leftcol.%s" % (filepath, args.extension),image)
    for idx, linecoords in enumerate(list_linecoords):
        # Header
        if idx == 0:
            if not args.quiet: print("header")
            roi = image[0:linecoords.height_start - 2, 0:image_param.width]  # region of interest
            roi = np.rot90(roi, 4 - args.horlinepos)
            with warnings.catch_warnings():
                # Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                if args.horlinetype == 1 and 'f' in args.croptypes:
                    pass
                elif 'h' in args.croptypes:
                    imgpath = "%s_TOC.%s" % (filepath, args.extension)
                    print(imgpath)
                    imsave(imgpath, roi)
                    return imgpath
    imgpath = "%s_TOC.%s" % (filepath, args.extension)
    imsave(imgpath, image)
    return imgpath

def deskew(args,image, image_param):
    # Deskew the given image based on the horizontal line
    # Calculate the angle of the points between 20% and 80% of the line
    uintimage = get_uintimg(image)
    binary = get_binary(args, uintimage)
    for x in range(0,args.binary_dilation):
        binary = ski.morphology.binary_dilation(binary,selem=np.ones((3, 3)))
    labels, numl = measurements.label(binary)
    objects = measurements.find_objects(labels)
    deskew_path = None
    for i, b in enumerate(objects):
        linecoords = Linecoords(image, i, b)
        # The line has to be bigger than minwidth, smaller than maxwidth, stay in the top (30%) of the img,
        # only one obj allowed and the line isn't allowed to start contact the topborder of the image
        if int(args.minwidthhor * image_param.width) < get_width(b) < int(args.maxwidthhor * image_param.width) \
                and int(image_param.height * args.minheighthor) < get_height(b) < int(image_param.height * args.maxheighthor) \
                and int(image_param.height * args.minheighthormask) < (linecoords.height_start+linecoords.height_stop)/2 < int(image_param.height * args.maxheighthormask) \
                and linecoords.height_start != 0:

            pixelwidth = set_pixelground(binary[b].shape[1])
            mean_y = []
            #Calculate the mean value for every y-array
            old_start = None
            for idx in range(pixelwidth(args.deskewlinesize)):
                value_y = measurements.find_objects(labels[b][:, idx + pixelwidth((1.0-args.deskewlinesize)/2)] == i + 1)[0]
                if old_start is None:
                    old_start = value_y[0].start
                if abs(value_y[0].start-old_start) < 5:
                    mean_y.append(value_y[0].start)
                    old_start = value_y[0].start
            polyfit_value = np.polyfit(list(range(0,len(mean_y))), mean_y, 1)
            deskewangle = np.arctan(polyfit_value[0]) * (360 / (2 * np.pi))
            args.ramp = True
            deskew_image = transform.rotate(image, deskewangle, mode="edge")
            create_dir(image_param.pathout+os.path.normcase("/deskew/"))
            deskew_path = "%s_deskew.%s" % (image_param.pathout+os.path.normcase("/deskew/")+image_param.name, args.extension)
            deskewinfo = open(image_param.pathout+os.path.normcase("/deskew/")+image_param.name + "_deskewangle.txt", "w")
            deskewinfo.write("Deskewangle:\t%f" % deskewangle)
            deskewinfo.close()
            image_param.deskewpath = deskew_path
            with warnings.catch_warnings():
                #Transform rotate convert the img to float and save convert it back
                warnings.simplefilter("ignore")
                misc.imsave(deskew_path, deskew_image)
            break
    return deskew_path

def get_binary(args, image):
    thresh = th.threshold_sauvola(image, args.threshwindow, args.threshweight)
    binary = image > thresh
    binary = 1 - binary  # inverse binary
    binary = np.rot90(binary, args.horlinepos)
    return binary

def get_height(s):
    return s[0].stop-s[0].start

def get_linecoords(s):
    return [[s[0].start,s[0].stop],[s[1].start,s[1].stop]]

def get_mindist(s,length):
    # Computes the min. distance to the border and cuts the smallest one in half
    d1 = s[1].start
    d2 = length - s[1].stop
    if d1 < d2:
        return d1-int(d1*0.5)
    else:
        return d2-int(d2*0.5)

def get_uintimg(image):
    if len(image.shape) > 2:
        uintimage = color.rgb2gray(copy.deepcopy(image))
    else:
        uintimage = copy.deepcopy(image)
    if uintimage.dtype == "float64":
        with warnings.catch_warnings():
            # Transform rotate convert the img to float and save convert it back
            warnings.simplefilter("ignore")
            uintimage = ski.img_as_uint(uintimage, force_copy=True)
    return uintimage

def get_width(s):
    return s[1].stop-s[1].start

def linecoords_analyse(args,origimg, image_param, clippingmask, get_toc=False):
    # Computes the clipping coords of the masks
    image = get_uintimg(origimg)
    origimg = np.rot90(origimg, args.horlinepos)
    binary = get_binary(args, image)
    labels, numl = measurements.label(binary)
    objects = measurements.find_objects(labels)
    count_height = 0
    count_width = 0
    pixelheight = set_pixelground(image_param.height)
    pixelwidth = set_pixelground(image_param.width)
    list_linecoords = []
    border = image_param.width
    topline_width_stop = image_param.height# Init list of linecoordinates the format is: [0]: width.start, width.stopt,
    # [1]:height.start, height.stop, [2]: Type of line [B = blank, L = vertical line]
    for i, b in enumerate(objects):
        # The line has to be bigger than minwidth, smaller than maxwidth, stay in the top (30%) of the img,
        # only one obj allowed and the line isn't allowed to start contact the topborder of the image
        linecoords = Linecoords(labels, i, b)
        if pixelwidth(0.8) <  get_width(b) < pixelwidth(args.maxwidthhor):
            print(b)
        if pixelwidth(args.minwidthhor) <  get_width(b) < pixelwidth(args.maxwidthhor) \
                and pixelheight(args.minheighthor) < get_height(b) < pixelheight(args.maxheighthor) \
                and pixelheight(args.minheighthormask) <  linecoords.height_stop < pixelheight(args.maxheighthormask) \
                and count_width == 0 \
                and linecoords.height_start != 0:
            # Distance Calculation - defining the clippingmask
            border = get_mindist(b, image_param.width)
            topline_width_stop = b[0].stop + 2 # Lowest Point of object + 2 Pixel
            if clippingmask.user is None:
                clippingmask.width_start = border
                clippingmask.width_stop = image_param.width - border
                clippingmask.height_start = copy.deepcopy(topline_width_stop)
                clippingmask.height_stop = 0
            # Get coordinats of the line
            labels[b][labels[b] == i + 1] = 0
            count_width += 1
            if get_toc:
                list_linecoords.append(copy.deepcopy(linecoords))
        if pixelheight(args.minheightver) < get_height(b) < pixelheight(args.maxheightver) \
                and pixelwidth(args.minwidthver) < get_width(b) < pixelwidth(args.maxwidthver) \
                and pixelwidth(args.minwidthvermask) < (linecoords.width_start+linecoords.width_stop)/2 < pixelwidth(args.maxwidthvermask) \
                and float(get_width(b))/float(get_height(b)) < args.maxgradientver:
            linecoords.segmenttype = 'L' # Defaultvalue for segmenttype 'P' for horizontal lines
            if count_height == 0:
                if b[0].start - topline_width_stop > pixelheight(args.minsizeblank+args.minsizeblankobolustop):
                    blankline = Linecoords(labels,i,b)
                    blankline.segmenttype = 'B'
                    blankline.height_start = topline_width_stop
                    blankline.height_stop = linecoords.height_start
                    blankline.width_start = border
                    blankline.width_stop = image_param.width - border
                    blankline.middle = int(((linecoords.width_start+linecoords.width_stop)-1)/2)
                    list_linecoords.append(copy.deepcopy(blankline))
                    count_height += 1
                    if args.ramp != None:
                        whiteout_ramp(origimg, linecoords)
                    list_linecoords.append(copy.deepcopy(linecoords))
                    count_height += 1
                else:
                    # Should fix to short vertical lines, in the height to top if they appear before any B Part in the image
                    if topline_width_stop > 0:
                        linecoords.height_start = topline_width_stop + pixelheight(args.addstartheightab)
                    list_linecoords.append(copy.deepcopy(linecoords))
                    count_height += 1
                    if args.ramp != None:
                        whiteout_ramp(origimg, linecoords)
            elif list_linecoords[count_height - 1].height_stop < b[0].stop:
                #Test argument to filter braces
                if b[0].start - list_linecoords[count_height - 1].height_stop > pixelheight(args.minsizeblank):
                    blankline = Linecoords(labels,i,b)
                    blankline.segmenttype = 'B'
                    blankline.height_start = list_linecoords[count_height - 1].height_stop
                    blankline.height_stop = linecoords.height_start
                    blankline.width_start = border
                    blankline.width_stop = image_param.width - border
                    blankline.middle = int(((linecoords.width_start+linecoords.width_stop)-1)/2)
                    list_linecoords.append(copy.deepcopy(blankline))
                    count_height += 1
                    list_linecoords.append(copy.deepcopy(linecoords))
                    if args.ramp != None:
                        whiteout_ramp(origimg, linecoords)
                    count_height += 1
                    labels[b][labels[b] == i + 1] = 0
                else:
                    if args.ramp != None:
                        whiteout_ramp(origimg, linecoords)
                    print(b[0].stop)
                    list_linecoords[count_height - 1].height_stop = b[0].stop
                    labels[b][labels[b] == i + 1] = 0
    return border, labels, list_linecoords, topline_width_stop

def set_pixelground(image_length):
    #Computes the real pixel number out of the given percantage
    def get_pixel(prc):
        return int(image_length*prc)
    return get_pixel

def whiteout_ramp(image, linecoords):
    # Dilation enlarge the bright segments and cut them out off the original image
    imagesection = image[linecoords.object]
    count = 0
    for i in morph.dilation(linecoords.object_matrix, morph.square(10)):
        whitevalue = measurements.find_objects(i == linecoords.object_value + 1)
        if whitevalue:
            whitevalue = whitevalue[0][0]
            imagesection[count,whitevalue.start:whitevalue.stop] = 255
            count +=1
    return 0

####################### MAIN-FUNCTIONS ############################################
def get_toc(args, imgpath=None):
    if imgpath is not None:
        args.input = imgpath
    lcol = cropping_lcol(args.input, args)
    if lcol is not None:
        tocpath = cropping_toc(lcol, args)
        return tocpath
    else:
        print("Left column was not found!")
        return ""
