import os
from subprocess import check_output, Popen, PIPE
from crop import get_toc
from bs4ocr import subtract_bg
import argparse
from pathlib import Path
import numpy as np

def get_parser():
    arg_parser = argparse.ArgumentParser(description='Extract table of contant from Reichsanzeiger.')
    arg_parser.add_argument("fpath", help="filename of text file or path to files", nargs='*')
    arg_parser.add_argument("--url_output", default="", type=str, help="Outpath if inputs are urls")
    arg_parser.add_argument("--inputfile", action="store_true",  help="The input is a file which contains one image/url per row")
    arg_parser.add_argument("--crop_only", action="store_true", help="Deactivate OCR")
    arg_parser.add_argument("--outputfolder", default="./cleaned", help="filename of the output")
    arg_parser.add_argument("--extensionaddon", default=".prep", help="Addon to the fileextension")
    arg_parser.add_argument("--blursize", default=59, type=int, help="Kernelsize for medianBlur")
    arg_parser.add_argument("--bluriter", default=1, type=int, help="Iteration of the medianBlur")
    arg_parser.add_argument("--fixblursize", action="store_true", help="Deactivate decreasing Blurkernelsize")
    arg_parser.add_argument("--blurfilter", default="Gaussian", type=str, help="Kernelsize for dilation",
                            choices=["Gaussian", "Median"])
    arg_parser.add_argument("--dilsize", default=5, type=int, help="Kernelsize for dilation")
    arg_parser.add_argument("--kernelshape", default="ellipse", type=str, help="Shape of the kernel for dilation",
                            choices=["cross", "ellipse", "rect"])
    arg_parser.add_argument("--contrast", default=0.0, type=float, help="Higher contrast (experimental)")
    arg_parser.add_argument("--normalize", action="store_true", help="Higher contrast (experimental)")
    arg_parser.add_argument("--normalize-only", action="store_true", help="Normalizes the image but doesnt subtract")
    arg_parser.add_argument("--normalize_auto", action="store_true", help="Auto-Normalization (experimental)")
    arg_parser.add_argument("--normalize_min", default=0, type=int, help="Min value for background normalization")
    arg_parser.add_argument("--normalize_max", default=255, type=int, help="Max value for background normalization")
    arg_parser.add_argument("--scale_channel", default="None", type=str, help="Shape of the kernel for dilation",
                            choices=["None", "red", "green", "blue", "cyan", "magenta", "yellow"])
    arg_parser.add_argument("--scale_channel_value", default=0.0, type=float, help="Scale value")
    arg_parser.add_argument("--binarize", action="store_true", help="Use Adaptive-Otsu-Binarization")
    arg_parser.add_argument("--dpi", default=300, type=int, help="Dots per inch (This value is used for binarization)")
    arg_parser.add_argument("--textdilation", action="store_false", help="Deactivate extra dilation for text")
    arg_parser.add_argument("--quality", default=100, help="Compress quality of the image like jpg")
    arg_parser.add_argument("-v", "--verbose", help="show ignored files", action="store_true")
    arg_parser.add_argument("--extension", type=str, choices=["bmp", "jpg", "png", "tif"], default="jpg",
                            help='Extension of the files, default: %(default)s')
    arg_parser.add_argument('-A', '--addstartheightab', type=float, default=0.01, choices=np.arange(-1.0, 1.0),
                            help='Add some pixel for the clipping mask of segments a&b (startheight), default: %(default)s')
    arg_parser.add_argument('-a', '--addstopheightab', type=float, default=0.011, choices=np.arange(-1.0, 1.0),
                            help='Add some pixel for the clipping mask of segments a&b (stopheight), default: %(default)s')
    arg_parser.add_argument('-C', '--addstartheightc', type=float, default=-0.005, choices=np.arange(-1.0, 1.0),
                            help='Add some pixel for the clipping mask of segment c (startheight), default: %(default)s')
    arg_parser.add_argument('-c', '--addstopheightc', type=float, default=0.0, choices=np.arange(-1.0, 1.0),
                            help='Add some pixel for the clipping mask of segment c (stopheight), default: %(default)s')
    arg_parser.add_argument('--bgcolor', type=int, default=1,
                            help='Backgroundcolor of the splice image (for "uint8": 0=black,...255=white): %(default)s')
    arg_parser.add_argument('--crop', action="store_false", help='cropping paper into segments')
    arg_parser.add_argument("--croptypes", type=str, nargs='+', choices=['a', 'b', 'c', 'f', 'h'],
                            default=['a', 'b', 'c', 'f', 'h'],
                            help='Types to be cropped out, default: %(default)s')
    arg_parser.add_argument("--binary_dilation", type=int, choices=[0, 1, 2, 3], default=0,
                            help='Dilate x-times the binarized areas.')
    arg_parser.add_argument("--horlinepos", type=int, choices=[0, 1, 2, 3], default=0,
                            help='Position of the horizontal line(0:top, 1:right,2:bottom,3:left), default: %(default)s')
    arg_parser.add_argument("--horlinetype", type=int, choices=[0, 1], default=0,
                            help='Type of the horizontal line (0:header, 1:footer), default: %(default)s')
    arg_parser.add_argument("--imgmask", type=float, nargs=4, default=[0.0, 1.0, 0.0, 1.0],
                            help='Set a mask that only a specific part of the image will be computed, arguments =  Heightstart, Heightend, Widthstart, Widthend')
    arg_parser.add_argument('--minwidthmask', type=float, default=0.06, choices=np.arange(0, 0.5),
                            help='min widthdistance of all masks, default: %(default)s')
    arg_parser.add_argument('--minwidthhor', type=float, default=0.55, choices=np.arange(0, 1.0),
                            help='minwidth of the horizontal lines, default: %(default)s')
    arg_parser.add_argument('--maxwidthhor', type=float, default=0.99, choices=np.arange(-1.0, 1.0),
                            help='maxwidth of the horizontal lines, default: %(default)s')
    arg_parser.add_argument('--minheighthor', type=float, default=0.00, choices=np.arange(0, 1.0),
                            help='minheight of the horizontal lines, default: %(default)s')
    arg_parser.add_argument('--maxheighthor', type=float, default=0.98, choices=np.arange(0, 1.0),
                            help='maxheight of the horizontal lines, default: %(default)s')
    arg_parser.add_argument('--minheighthormask', type=float, default=0.04, choices=np.arange(0, 1.0),
                            help='minheight of the horizontal lines mask (search area), default: %(default)s')
    arg_parser.add_argument('--maxheighthormask', type=float, default=0.99, choices=np.arange(0, 1.0),
                            help='maxheight of the horizontal lines mask (search area), default: %(default)s')
    arg_parser.add_argument('--minheightver', type=float, default=0.0375, choices=np.arange(0, 1.0),
                            help='minheight of the vertical lines, default: %(default)s')  # Value of 0.035 is tested (before 0.05)
    arg_parser.add_argument('--maxheightver', type=float, default=0.95, choices=np.arange(0, 1.0),
                            help='maxheightof the vertical lines, default: %(default)s')
    arg_parser.add_argument('--minwidthver', type=float, default=0.00, choices=np.arange(0, 1.0),
                            help='minwidth of the vertical lines, default: %(default)s')
    arg_parser.add_argument('--maxwidthver', type=float, default=0.022, choices=np.arange(0, 1.0),
                            help='maxwidth of the vertical lines, default: %(default)s')
    arg_parser.add_argument('--minwidthvermask', type=float, default=0.1, choices=np.arange(0, 1.0),
                            help='minwidth of the vertical lines mask (search area), default: %(default)s')
    arg_parser.add_argument('--maxwidthvermask', type=float, default=0.4, choices=np.arange(0, 1.0),
                            help='maxwidth of the vertical lines mask (search area), default: %(default)s')
    arg_parser.add_argument('--maxgradientver', type=float, default=0.05, choices=np.arange(0, 1.0),
                            help='max gradient of the vertical lines: %(default)s')
    # 0.016
    arg_parser.add_argument('--minsizeblank', type=float, default=0.015, choices=np.arange(0, 1.0),
                            help='min size of the blank area between to vertical lines, default: %(default)s')
    arg_parser.add_argument('--minsizeblankobolustop', type=float, default=0.014, choices=np.arange(0, 1.0),
                            help='min size of the blank area between to vertical lines, default: %(default)s')
    arg_parser.add_argument('--nomnumber', type=int, default=4,
                            help='Sets the quantity of numbers in the nomenclature (for "4": 000x_imagename): %(default)s')
    arg_parser.add_argument('--parallel', type=int, default=1, help="number of CPUs to use, default: %(default)s")
    arg_parser.add_argument('--ramp', default=None, help='activates the function whiteout')
    arg_parser.add_argument('--adaptingmasksoff', action="store_true", help='deactivates adapting maskalgorithm')
    arg_parser.add_argument('--showmasks', action="store_false", help='output an image with colored masks')
    arg_parser.add_argument('--specialnomoff', action="store_false",
                            help='Disable the special nomenclature for the AKF-Project!')
    arg_parser.add_argument('--splice', action="store_false", help='splice the cropped segments')
    arg_parser.add_argument("--splicetypes", type=str, nargs='+', choices=['a', 'b', 'c', 'f', 'h'],
                            default=['a', 'b', 'c'],
                            help='Segmenttypes to be spliced, default: %(default)s')
    arg_parser.add_argument("--splicemaintype", type=str, choices=['a', 'b', 'c', 'f', 'h'], default='c',
                            help='Segmenttype that indicates a new splice process, default: %(default)s')

    arg_parser.add_argument('--splicemaintypestop', action="store_true",
                            help='The maintype of splicetyps will be placed on the end')
    arg_parser.add_argument('--threshwindow', type=int, default=31,
                            help='Size of the window (binarization): %(default)s')
    arg_parser.add_argument('--threshweight', type=float, default=0.2, choices=np.arange(0, 1.0),
                            help='Weight the effect of the standard deviation (binarization): %(default)s')
    arg_parser.add_argument('--woblankstop', action="store_true",
                            help='Deactivates the whiteout of the blank parts for the a & b parts, this will lead to less memory usage.')
    arg_parser.add_argument('-q', '--quiet', action='store_true', help='be less verbose, default: %(default)s')
    args = arg_parser.parse_args()
    return args


def extract_toc():
    import logging
    logging.basicConfig(filename='ratocr.log', level=logging.ERROR)
    args = get_parser()
    if args.inputfile and os.path.isfile(args.fpath[0]):
        with open(args.fpath[0],"r") as fin:
            args.fpath = fin.read().splitlines()
    elif len(args.fpath) == 1 and not os.path.isfile(args.fpath[0]) and args.url_output == "":
        args.fpath = list(Path(args.fpath[0]).rglob(f"*.jpg"))
    for imgpath in args.fpath:
        # imgpath = '/home/jkamlah/Documents/Reichsanzeiger/First_pages_RA/856399094_1931_001_01.jpg'
        # If the fpath is an url download the image
        print(f"Start processing: {imgpath}")
        ipath = imgpath
        try:
            if args.url_output != "":
                outputpath = Path(args.url_output)
                foutpath = outputpath.joinpath(os.path.join("TOC-Extraction",os.path.basename(imgpath).rsplit(".",1)[0]))
                if not foutpath.exists():
                    foutpath.mkdir(parents=True, exist_ok=True)
                    foutpath.joinpath('url.info').write_text(imgpath)
                elif list(foutpath.rglob("*.txt")):
                    continue
                import requests
                import shutil
                r = requests.get(str(imgpath), stream=True)
                ipath = imgpath
                imgpath = outputpath.joinpath(os.path.basename(imgpath))
                if r.status_code == 200:
                    with open(imgpath, 'wb') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)
                else:
                    continue
            imgpath = Path(imgpath)
            # Extract TOC
            tocpath = get_toc(args, imgpath)
            # Substract Background
            tocpath_clean = subtract_bg(args, tocpath)
            # OCR TOC
            if not args.crop_only:
                ocr_text = check_output(["tesseract",
                                         "-l",
                                         "frak2021",
                                         tocpath_clean,
                                         "stdout"], universal_newlines=True)
            if not isinstance(tocpath, str):
                tocpath = str(tocpath.absolute())
            with open(tocpath.rsplit(".", 1)[0] + ".txt", "w") as fout:
                fout.write(ocr_text.strip())
        except:
            logging.error(f"{ipath} was not processed properly.")
        if args.url_output != "":
            imgpath.unlink()


if __name__ == "__main__":
    extract_toc()
