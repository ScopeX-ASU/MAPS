import cv2
import gdspy
import matplotlib.image as img
import numpy as np
from prefab.processor import *


def dev2cell(
    library, device, res, cell_name, layer=1, cntr_type=cv2.CHAIN_APPROX_SIMPLE
):
    # flip image (has to be done)
    device = np.flipud(device)

    # get device contours
    _, thresh = cv2.threshold(device.astype(np.uint8), 0.5, 1, 0)
    contours = cv2.findContours(thresh, cv2.RETR_CCOMP, cntr_type)

    # add polygon(s) to cell
    outers = []
    inners = []
    for idx, contour in enumerate(contours[0]):
        if len(contour) > 2:
            contour = contour / 1000  # μm to nm
            points = contour.squeeze().tolist()
            points = list(tuple(sub) for sub in points)
            if contours[1][0][idx][3] == -1:
                outers.append(points)
            else:
                inners.append(points)
    cell = library.new_cell(cell_name)
    poly = gdspy.boolean(outers, inners, "xor", layer=layer)
    poly = poly.scale(res, res)
    cell.add(poly)
    return gdspy.CellReference(cell)


def load_device_img(path, slice_length, device_length, res):
    return _autoscale(img.imread(path)[:, :, 1], slice_length, device_length, res)


def load_device_gds(path, cell_name, slice_length, device_length, res):
    gds = gdspy.GdsLibrary(infile=path)
    cell = gds.cells[cell_name]
    polygons = cell.get_polygons()
    dims = cell.get_bounding_box()
    device = np.zeros((int(dims[1][0] * 1000), int(dims[1][1] * 1000)))

    # needs a better method
    contours = []
    for k in range(len(polygons)):
        contour = []
        for j in range(len(polygons[k])):
            contour.append(
                [[int(1000 * polygons[k][j][0]), int(1000 * polygons[k][j][1])]]
            )
        contours.append(np.array(contour))

    cv2.drawContours(device, contours, -1, (255, 255, 255), -1)
    device = trim(np.flipud(device))
    return _autoscale(device, slice_length, device_length, res)


def _autoscale(device, slice_length, device_length, res):
    # scale the device so that res*length_px = length_nm
    # units are still px, but different
    device = trim(device)
    scale = 1 / (res / (device_length / device.shape[1]))
    device = cv2.resize(device, (0, 0), fx=scale, fy=scale)
    device = binarize(device)

    # pad to multiple of slice_length
    device = pad(device, slice_length=slice_length, fctr=2)
    return device
