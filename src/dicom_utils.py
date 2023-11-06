# -*- coding: utf-8 -*-
"""
Code written by Carlos E. Cardenas 9.21.2018
email: cecardenas@mdanderson.org

Do not distribute without permission
"""

import os, glob, pickle

import SimpleITK as sitk
import pydicom as dicom
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from skimage import morphology
import numpy as np

import matplotlib.pyplot as plt
from skimage.draw import polygon
import skimage.morphology as skmo
import skimage.measure as skme
import skimage.transform as sktm
import cv2

import logging, time, datetime

# change
def read_images_labels(path, im_tag='CT', rs_tag='RTSTRUCT', ds_tag='RTDOSE', plan_tag='RTPLAN', rois=None, verbose=False):
    """Reads the images and labels from a folder containing both dicom files

    Args:
        path:           Path containing all DICOM files
        im_tag:         DICOM image file name tag (first letters for this file type)
        rs_tag:         DICOM RT Structure file name tag (first letters for this file type)
        ds_tag:         DICOM RT Dose file name tag (first letters for this file type)
        plan_tag:       DICOM RT Plan file name tag (first letters for this file type)
        rois:           Array of string containing defined rois to extract (i.e. ['Mandible', 'Cord']). Choosing 'None'
                        imports all rois in the DICOM RT Structure files located in 'path'.
    Returns:
        image:          Numpy array (slice, row, column) containing the appended slices (from each DICOM image file)
                        to create a 3D array.
        dicom_info:     Dictionary containing the DICOM information for each slice corresponding to 'image'
        contours:       List of dictionaries containing roi points and information for each roi in the list.
        dose:           CURRENTLY EMPTY
        plan_info:      CURRENTLY EMPTY

    To do:
        Add component to read RTDOSE files and convert them to numpy arrays.
        Add component to read RTPLAN files.

    """

    image = []
    dicom_info = []
    contours = []
    dose = []
    plan_info = []

    if im_tag is None:
        im_tag = ''
    if rs_tag is None:
        rs_tag = ''
    if ds_tag is None:
        ds_tag = ''
    if plan_tag is None:
        plan_tag = ''

    if any([im_tag == rs_tag, im_tag == plan_tag, im_tag == ds_tag, rs_tag == plan_tag, rs_tag == ds_tag,
            ds_tag == plan_tag]):

        start = time.time()
        dcm_paths = []
        for subdir, dirs, files in os.walk(path):
            dcms = glob.glob(os.path.join(subdir, im_tag + '*.dcm'))
            if len(dcms) > 0:
                dcm_paths.extend(dcms)
            if not rs_tag == None:
                dcms = glob.glob(os.path.join(subdir, rs_tag + '*.dcm'))
                if len(dcms) > 0:
                    dcm_paths.extend(dcms)
            if not ds_tag == None:
                dcms = glob.glob(os.path.join(subdir, ds_tag + '*.dcm'))
                if len(dcms) > 0:
                    dcm_paths.extend(dcms)
            if not plan_tag == None:
                dcms = glob.glob(os.path.join(subdir, plan_tag + '*.dcm'))
                if len(dcms) > 0:
                    dcm_paths.extend(dcms)

        dcm_paths = list(set(dcm_paths))
        # print('elapsed time getting file paths: ' + str(time.time()-start))
        start = time.time()
        for dcm_path in dcm_paths:
            dcm_info = dicom.read_file(dcm_path)
            mod = dcm_info.Modality
            if mod == 'CT' or mod == 'MR' or mod == 'PT':
                try:
                    hi = float(dcm_info.ImagePositionPatient[2])
                except:
                    continue
                dicom_info.append(dcm_info)
            elif mod == "RTSTRUCT":
                if not rs_tag == None:
                    rs_start = time.time()
                    tmp = read_contours(dicom_dataset=dcm_info, rois=rois)
                    # print('elapsed time reading contours: ' + str(time.time()-rs_start))
                    contours.extend(tmp)
            elif mod == "RTPLAN":
                if not plan_tag == None:
                    plan_info.append(dicom.read_file(dcm_path))
            elif mod == "RTDOSE":
                if not ds_tag == None:
                    dose.append(dicom.read_file(dcm_path))

        # print('elapsed time loading dicoms: ' + str(time.time()-start))
        dicom_info.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        image = np.stack([s.pixel_array for s in dicom_info], axis=0).astype(np.float32)
        image = image + dicom_info[0].RescaleIntercept

    else:
        for subdir, dirs, files in os.walk(path):
            dcms = glob.glob(os.path.join(subdir, im_tag+'*.dcm'))
            if len(dcms) > 1:
                dicom_info = [dicom.read_file(dcm) for dcm in dcms]
                dicom_info.sort(key=lambda x: float(x.ImagePositionPatient[2]))
                image = np.stack([s.pixel_array for s in dicom_info], axis=0).astype(np.float32)
                if dicom_info[0].Modality == 'CT':
                    image = image + dicom_info[0].RescaleIntercept
            else:
                if verbose:
                    print('  Missing image files with tag: ' + im_tag)

            if not rs_tag == None:
                dcms = glob.glob(os.path.join(subdir, rs_tag+'*.dcm'))
                if len(dcms) >= 1:
                    for dcm in dcms:
                        tmp = read_contours(dcm, rois=rois)
                        contours.extend(tmp)
                else:
                    if verbose:
                        print('  Missing RS file with tag: ' + rs_tag)
            if not ds_tag == None:
                dcms = glob.glob(os.path.join(subdir, ds_tag + '*.dcm'))
                if len(dcms) >= 1:
                    for dcm in dcms:
                        dose.append(dicom.read_file(dcm))

            if not plan_tag == None:
                dcms = glob.glob(os.path.join(subdir, plan_tag + '*.dcm'))
                if len(dcms) >= 1:
                    for dcm in dcms:
                        plan_info.append(dicom.read_file(dcm))

    return image, dicom_info, contours, dose, plan_info


def read_contours(dicom_path=None, dicom_dataset=None, rois=None):
    """ Reads the DICOM RT Structure file and outputs a list of dictionaries with roi points and relevant information
        for each roi.

    Args:
        dicom_path:     Path containing the DICOM RT Structure file
        dicom_dataset:  Dicom Dataset from pydicom
        rois:           Array of string containing defined rois to extract (i.e. ['Mandible', 'Cord']).
                        Choosing 'None' imports all rois in the DICOM RT Structure files located in 'dicom_path'.
    Returns:
        contours:       List of dictionaries containing roi points and information for each roi in the list.

    """
    contours = []
    if dicom_dataset is None and dicom_path is not None:
        structure = dicom.read_file(dicom_path)
    elif dicom_dataset is not None and dicom_path is None:
        structure = dicom_dataset
    else:
        print('Either a dicom file path or a pydicom dataset are required to read contours')
        return contours

    for i in range(len(structure.ROIContourSequence)):
        if rois is None:
            contour = {}
            try:
                contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
            except AttributeError:
                contour['color'] = dicom.multival.MultiValue(dicom.valuerep.IS, ['0', '0', '0'])
            contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
            contour['name'] = structure.StructureSetROISequence[i].ROIName
            assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
            try:
                contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
            except AttributeError:
                contour['contours'] = []
                print('   Missing contours for structure: ' + contour['name'])
            contours.append(contour)
        else:
            for roi in rois:
                if roi == structure.StructureSetROISequence[i].ROIName:
                    contour = {}
                    try:
                        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
                    except AttributeError:
                        contour['color'] = dicom.multival.MultiValue(dicom.valuerep.IS, ['0', '0', '0'])
                    contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
                    contour['name'] = structure.StructureSetROISequence[i].ROIName
                    assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
                    try:
                        contour['contours'] = []
                        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
                    except AttributeError:
                        print('   Missing contours for structure: ' + contour['name'])
                    contours.append(contour)
                    break
    return contours


def get_labels(contours, shape, dicom_info, rois=None):
    """ Uses the outputs from the function "read_images_labels" to convert the contours to boolean numpy arrays.

    Args:
        contours:       List of dictionaries containing roi points and information for each roi in the list.
                        This is one of the outputs from the function "read_images_labels"
        shape:          Tuple. Shape of the 3D image for which the contours were delineated on.
        dicom_info:     Dictionary containing the DICOM information for each slice corresponding to the 3D image
        rois:           Array of string containing defined rois to extract (i.e. ['Mandible', 'Cord']).
                        Choosing 'None' imports all rois found in 'contours'.

    Returns:
        label_map:      Boolean numpy array of shape equal to 'shape' for the corresponding rois found in 'contours'.

    """
    if not contours == []:

        z = [np.around(s.ImagePositionPatient[2], 1) for s in dicom_info]
        pos_r = dicom_info[0].ImagePositionPatient[1] * dicom_info[0].ImageOrientationPatient[0]
        spacing_r = dicom_info[0].PixelSpacing[1] 
        pos_c = dicom_info[0].ImagePositionPatient[0] * dicom_info[0].ImageOrientationPatient[0]
        spacing_c = dicom_info[0].PixelSpacing[0]



        if rois is None:
            label_map = np.zeros([shape[0], shape[1], shape[2], len(contours)], dtype=np.bool)
            num = 1
            ct = 0
            for con in contours:
                for c in con['contours']:
                    contoured_points = np.array(c).reshape((-1, 3))
                    if contoured_points.shape[0] > 1:
                        assert np.amax(np.abs(np.diff(contoured_points[:, 2]))) == 0
                        try:
                            z_index = z.index(np.around(contoured_points[0, 2], 1))
                        except ValueError:
                            print("   Missing CT slice")
                            break
                        r = (contoured_points[:, 1] - pos_r) / spacing_r
                        c = (contoured_points[:, 0] - pos_c) / spacing_c
                        rr, cc = polygon(r, c)
                        if any(rr > 511) or any(rr < 0) or any(cc < 0) or any(cc > 511):
                            rr[rr > 511] = label_map.shape[1]-1
                            rr[rr < 0] = 0
                            cc[cc > 511] = label_map.shape[2]-1
                            cc[cc < 0] = 0
                        label_map[z_index, rr, cc, ct] = True
                        if dicom_info[0].ImageOrientationPatient[0] == -1:
                            label_map[z_index, :, :, ct] = np.flipud(label_map[z_index, :, :, ct])
                ct += 1
        else:
            if type(rois) == list:
                if len(rois) <= 1:
                    label_map = np.zeros([shape[0], shape[1], shape[2]], dtype=np.bool)
                else:
                    label_map = np.zeros([shape[0], shape[1], shape[2], len(rois)], dtype=np.bool)
            elif type(rois) == str:
                label_map = np.zeros([shape[0], shape[1], shape[2]], dtype=np.bool)

            # finding match between contour_names and roi_names
            names_in_contours = []
            for c in contours:
                names_in_contours.append(c['name'])

            idx = []
            if type(rois) == list:
                for roi in rois:
                    ct = 0
                    found = False
                    for name in names_in_contours:
                        if name == roi:
                            idx.append(ct)
                            found = True
                            break
                        else:
                            ct += 1
                    if found == False:
                        idx.append(-1)
            elif type(rois) == str:
                idx = []
                found = False
                for ct, name in enumerate(names_in_contours):
                    if name == rois:
                        idx.append(ct)
                        found = True
                        break
                if found == False:
                    idx.append(-1)

            for i, index in enumerate(idx):
                if index >= 0:
                    num = 1
                    con = contours[index]
                    for c in con['contours']:
                        nodes = np.array(c).reshape((-1, 3))
                        if nodes.shape[0] > 1:
                            assert np.amax(np.abs(np.diff(nodes[:, 2]))) < 0.0001
                            try:
                                z_index = z.index(np.around(nodes[0, 2], 1))
                            except ValueError:
                                print("   Missing CT slice")
                                break
                            r = (nodes[:, 1] - pos_r) / spacing_r
                            c = (nodes[:, 0] - pos_c) / spacing_c
                            rr, cc = polygon(r, c)
                            if any(rr > 511) or any(rr < 0) or any(cc < 0) or any(cc > 511):
                                rr[rr > 511] = label_map.shape[1] - 1
                                rr[rr < 0] = 0
                                cc[cc > 511] = label_map.shape[2] - 1
                                cc[cc < 0] = 0
                            if type(rois) == list:
                                if len(rois) <= 1:
                                    label_map[z_index, rr, cc] = num
                                else:
                                    label_map[z_index, rr, cc, i] = num
                            elif type(rois) == str:
                                label_map[z_index, rr, cc] = num
        label_map = label_map.astype(np.bool)
    else:
        label_map = []
    return label_map

def RTStruct(maskVolumeStruct, maskName, color, DICOMImageStruct, fname='RS_test.dcm', flag_pos=0, flag_ori=0, flag_resize_xy=0, resize_shift=[0.,0.], ref_pixsize=[2.500, 0.9765625, 0.9765625]):
    logging.info("Writing the RT Structure file...")
    # Create , dicom structure
    ds = writeDCMheader(fname, DICOMImageStruct[0])

    # Determine Orientation
    singleSliceDis = [abs(DICOMImageStruct[1].ImagePositionPatient[0] - DICOMImageStruct[0].ImagePositionPatient[0]),
                      abs(DICOMImageStruct[1].ImagePositionPatient[1] - DICOMImageStruct[0].ImagePositionPatient[1]),
                      abs(DICOMImageStruct[1].ImagePositionPatient[2] - DICOMImageStruct[0].ImagePositionPatient[2])]

    index = singleSliceDis.index(max(singleSliceDis))
    logging.info("index : {}".format(index))

    if index == 0:
        orientation = 'Sagittal'
        a = 1
        b = 2
        c = 0
    elif index == 1:
        orientation = 'Coronal'
        a = 0
        b = 2
        c = 1
    else:
        orientation = 'Axial'
        a = 0
        b = 1
        c = 2

    # get Slice position of start and end
    slicePos = np.zeros(len(DICOMImageStruct))
    for i in range(len(DICOMImageStruct)):
        slicePos[i] = DICOMImageStruct[i].ImagePositionPatient[index]

    # slice thickness and spacing of DICOM image
    zspc_img = DICOMImageStruct[0].SliceThickness
    xspc_img = DICOMImageStruct[0].PixelSpacing[0]
    yspc_img = DICOMImageStruct[0].PixelSpacing[1]

    if flag_resize_xy == 1:
        xspc_img = ref_pixsize[1]
        yspc_img = ref_pixsize[2]

    # slice thickness and spacing for Mask
    image_numel_x = DICOMImageStruct[0].Rows
    image_numel_y = DICOMImageStruct[0].Columns

    imagepos_x = DICOMImageStruct[0].ImagePositionPatient[a]
    imagepos_y = DICOMImageStruct[0].ImagePositionPatient[b]

    logging.info(DICOMImageStruct[0].ImagePositionPatient)

    numROI = len(maskVolumeStruct) # maskVolumeStruct dim : in numROI number of list, ([z, x, y])

    ContourGeometricType = 'CLOSED_PLANAR'

    ROIContourSequence = Sequence()
    StructureSetROISequence = Sequence()
    RTROIObservationsSequence = Sequence()

    ReferencedFrameOfReferenceUID = DICOMImageStruct[0].FrameOfReferenceUID


    #################### ReferencedFrameOfReferenceSequence ####################
    ContourImageSequence1 = Sequence()
    logging.info("LENGTH DICOM IMAGE STRUCT : {}".format(len(DICOMImageStruct)))
    for i in range(len(DICOMImageStruct)):
        item1 = Dataset()
        item1.ReferencedSOPClassUID = DICOMImageStruct[i].SOPClassUID
        item1.ReferencedSOPInstanceUID = DICOMImageStruct[i].SOPInstanceUID

        ContourImageSequence1.append(item1)

    item2 = Dataset()
    item2.SeriesInstanceUID = DICOMImageStruct[0].SeriesInstanceUID
    item2.ContourImageSequence = ContourImageSequence1

    item3 = Dataset()
    RTReferencedSeriesSequence = Sequence()
    RTReferencedSeriesSequence.append(item2)

    item3.RTReferencedSeriesSequence = RTReferencedSeriesSequence
    item3.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2'
    item3.ReferencedSOPInstanceUID = DICOMImageStruct[0].StudyInstanceUID

    item4 = Dataset()
    RTReferencedStudySequence = Sequence()
    RTReferencedStudySequence.append(item3)

    item4.RTReferencedStudySequence = RTReferencedStudySequence
    item4.FrameOfReferenceUID = DICOMImageStruct[0].FrameOfReferenceUID

    ReferencedFrameOfReferenceSequence = Sequence()
    ReferencedFrameOfReferenceSequence.append(item4)
    
    def color2value(color):
        if color=="red":
            colorVal = [255, 0, 0]
        elif color == "purple":
            colorVal = [255, 0, 255]
        elif color == "yellow":
            colorVal = [255,255,0]
        elif color == "blue":
            colorVal = [0,0,255]
        elif color == "orange":
            colorVal = [255,165,0]
        elif color == "green":
            colorVal = [0,255,0]
        return colorVal

    ########################################################################
    for nm in range(numROI):
        # ROIContourSequence
        structure_ds = Dataset()
        structure_ds.ROIDisplayColor = color2value(color[nm])
        structure_ds.ReferencedROINumber = nm + 1
        ContourSequence = Sequence()

        # StructureSetROISequence
        roi_ds = Dataset()
        roi_ds.ROINumber = nm + 1
        roi_ds.ROIName = maskName[nm]
        roi_ds.ROIGenerationAlgorithm = 'AUTOMATIC'
        roi_ds.ReferencedFrameOfReferenceUID = ReferencedFrameOfReferenceUID

        # RTROIObservationsSequence
        roiobs_ds = Dataset()
        roiobs_ds.ObservationNumber = nm + 1
        roiobs_ds.ReferencedROINumber = nm + 1
        roiobs_ds.RTROIInterpretedType = 'ORGAN'
        roiobs_ds.ROIInterpreter = 'AUTOMATIC'
        roiobs_ds.ROIObservationLabel = maskName[nm]

        MaskVolume = maskVolumeStruct[nm]
        MaskVolume = np.ascontiguousarray(np.squeeze(MaskVolume))

        """
        # If the position was not HFS
        if flag_pos == 1: # HFP
            for i in range(MaskVolume.shape[0]):
                MaskVolume[i] = np.flipud(MaskVolume[i])
        #elif flag_pos == 2: # FFS
        """

        # MaskVolume dim : [z, x, y]
        mask_numel_x = MaskVolume.shape[1]
        mask_numel_y = MaskVolume.shape[2]

        xratio = float(image_numel_x) / float(mask_numel_x)
        yratio = float(image_numel_y) / float(mask_numel_y)

        xspc = xspc_img * xratio
        yspc = yspc_img * yratio

        # Shift ImagePatientPosition to a new pixel arrangement
        # ASSUMPTION : The upper left corner of the upper left pixel must be the same before and after "sktf.resize"
        maskpos_x = imagepos_x - 0.5*xspc_img + 0.5*xspc
        maskpos_y = imagepos_y - 0.5*yspc_img + 0.5*yspc
        
        if flag_resize_xy == 1:
            maskpos_x = maskpos_x + resize_shift[0]
            maskpos_y = maskpos_y + resize_shift[1]

        # Postprocess
        MaskVolume = np.round(MaskVolume).astype(np.uint8)
        if maskName[nm] == 'Shoulders':
            num_volume = 2
        else:
            num_volume = 1

        MaskVolume = postprocess(MaskVolume.astype(bool), num_volume=num_volume)

        logging.info("Mask Name : {} -----------------------------".format(maskName[nm]))

        for numz in range(MaskVolume.shape[0]): # for different z-axis, 
            MaskVolume_temp = MaskVolume[numz]

            current_shape = MaskVolume_temp.shape

            contours, hierarchy = cv2.findContours(MaskVolume_temp.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            #contours, hierarchy = cv2.findContours(MaskVolume_temp.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

            # Extra step for outside contours
            if contours:
                temp_contours = []
                for part_contour in contours:
                    """
                    rl_list = [] # right and left one pixel case
                    rl_rest = []
                    ud_list = [] # up and down one pixel case
                    ud_rest = []
                    """

                    new_contours = []
                    for index, i in enumerate(part_contour):
                        xcoord = round(i[0][1],6)
                        ycoord = round(i[0][0],6)
                        
                        if xcoord > 1 and ycoord > 1 and xcoord < MaskVolume_temp.shape[0]-1 and ycoord < MaskVolume_temp.shape[1]-1:

                            up = MaskVolume_temp[xcoord,ycoord+1]
                            down = MaskVolume_temp[xcoord,ycoord-1]
                            left = MaskVolume_temp[xcoord-1,ycoord]
                            right = MaskVolume_temp[xcoord+1,ycoord]
                            
                            if up == False and right == False and down == True and left == True:
                                new_contours.append([[ycoord, xcoord+0.5]])
                                new_contours.append([[ycoord+0.5, xcoord]])
                            elif up == False and right == False and down == True and left == False:
                                new_contours.append([[ycoord, xcoord+0.5]])
                                new_contours.append([[ycoord+0.5, xcoord]])
                                new_contours.append([[ycoord, xcoord-0.5]])
                            elif up == False and right == False and down == False and left == True:
                                new_contours.append([[ycoord-0.5, xcoord]])
                                new_contours.append([[ycoord, xcoord+0.5]])
                                new_contours.append([[ycoord+0.5, xcoord]])
                            elif up == False and right == True and down == False and left == True:
                                hi = 0
                                #logging.info("One Pixel Case, thin horizontally")
                            elif up == True and right == False and down == True and left == False:
                                hi = 0
                                #logging.info("One Pixel Case, thin vertically")
                            else:
                                if ycoord < current_shape[1] and up == False:
                                    new_contours.append([[ycoord+0.5, xcoord]])
                                if xcoord > 0 and left == False:
                                    new_contours.append([[ycoord, xcoord-0.5]])
                                if ycoord > 0 and down == False:
                                    new_contours.append([[ycoord-0.5, xcoord]])
                                if xcoord < current_shape[1] and right == False:
                                    new_contours.append([[ycoord, xcoord+0.5]])
                        else:
                            new_contours.append([[ycoord, xcoord]])
                        
                        """ Better not having this part in terms of smoothness ...
                        # right left True case
                        elif up == False and right == True and down == False and left == True:
                            temp = [ycoord, xcoord]
                            left_coord = [ycoord, xcoord-0.5]
                            right_coord = [ycoord, xcoord+0.5]
                            dist1 = distance(new_contours[-1][0], left_coord)
                            dist2 = distance(new_contours[-1][0], right_coord)

                            if temp in rl_list:
                                index = rl_list.index(temp)
                                new_contours.append(rl_rest[index])
                            else:
                                rl_list.append(temp)
                                
                                # right is closer than left from the previous point
                                if dist1 > dist2: 
                                    new_contours.append([right_coord])
                                    rl_rest.append([left_coord])
                                else:
                                    new_contours.append([left_coord])
                                    rl_rest.append([right_coord])
                        # up down True case 
                        elif up == True and right == False and down == True and left == False:
                            temp = [ycoord, xcoord]
                            up_coord = [ycoord+0.5, xcoord]
                            down_coord = [ycoord-0.5, xcoord]
                            dist1 = distance(new_contours[-1][0], up_coord)
                            dist2 = distance(new_contours[-1][0], down_coord)
                            
                            if temp in ud_list:
                                index = ud_list.index(temp)
                                new_contours.append(ud_rest[index])
                            else:
                                ud_list.append(temp)
                                
                                # down is closer than up from the previous point
                                if dist1 > dist2: 
                                    new_contours.append([down_coord])
                                    ud_rest.append([up_coord])
                                else:
                                    new_contours.append([up_coord])
                                    ud_rest.append([down_coord])
                        """
                        
                    # remove duplicate
                    temp_new_contours = []
                    for kk in new_contours:
                        if kk not in temp_new_contours:
                            temp_new_contours.append(kk)
                        else:
                            hi = 0
                            #logging.info('{} removed'.format(kk))
                    
                    temp_contours.append(np.array(temp_new_contours))
                contours = temp_contours
            
            zpos = slicePos[numz]

            total_contours = np.array([])
            for i, temp_contours in enumerate(contours): # for multiple (separate) structures in a single slice
                temp_contours = np.squeeze(temp_contours)
                zlist = zpos * np.ones([temp_contours.shape[0],1])
                
                # Unless mask is too small to have one or two dots in contours
                if len(temp_contours.shape) > 1:
                    ###### Indent <<
                    total_contours = np.concatenate((temp_contours, zlist), axis=1)

                    if total_contours.shape[0] > 0:
                        total_contours = total_contours * np.array([xspc, yspc, 1]) + np.array([maskpos_x, maskpos_y, 0])
                        # Smoothing process using B-spline
                        #total_contours = bspline_smoothing(total_contours)

                        """
                        ################################# ImageOrientation ########################################
                        if flag_ori != 0:
                            for qq in range(total_contours.shape[0]):
                                #NEED TO UPDATE WHAT MAT IS!!!!!
                                #total_contours[qq,0] = mat[0]*total_contours[qq,0] + mat[3]*total_contours[qq,1]
                                #total_contours[qq,1] = mat[1]*total_contours[qq,0] + mat[4]*total_contours[qq,1]
                                
                                total_contours[qq,0] = -1*total_contours[qq,0]
                                total_contours[qq,1] = -1*total_contours[qq,1]
                        ###########################################################################################
                        """

                        ContourData = list(total_contours.flatten())

                        if len(ContourData) % 3 != 0:
                            logging.info(len(ContourData), " is not multiple of 3!!")
                            break

                        NumberOfContourPoints = int(len(ContourData) / 3)

                        slice_ds = Dataset()
                        slice_ds.ContourData = ContourData
                        slice_ds.NumberOfContourPoints = NumberOfContourPoints
                        slice_ds.ContourGeometricType = ContourGeometricType
            
                        # ROIContourSequence
                        # This should be written every slice, due to ReferencedSOPInstanceUID
                        ContourImageSequence = Sequence()
                        temp_ImageSequence = Dataset()
                        temp_ImageSequence.ReferencedSOPClassUID = DICOMImageStruct[0].SOPClassUID
                        minPos = abs(slicePos - zpos)
                        minInd = minPos.tolist().index(min(minPos))
                        #print("minInd : ", i, ", ", minInd)
                        temp_ImageSequence.ReferencedSOPInstanceUID = DICOMImageStruct[minInd].SOPInstanceUID
                        ContourImageSequence.append(temp_ImageSequence)
                        
                        slice_ds.ContourImageSequence = ContourImageSequence

                        ContourSequence.append(slice_ds)
                    ###### Indent <<
                else:
                    logging.info(temp_contours, temp_contours.shape, temp_contours.dtype)
                    logging.info(maskName[nm] + " is too small to contour for the current slice. Please check it!")

        structure_ds.ContourSequence = ContourSequence

        ROIContourSequence.append(structure_ds)
        StructureSetROISequence.append(roi_ds)
        RTROIObservationsSequence.append(roiobs_ds)

    ds.ReferencedFrameOfReferenceSequence = ReferencedFrameOfReferenceSequence
    ds.ROIContourSequence = ROIContourSequence
    ds.StructureSetROISequence = StructureSetROISequence
    ds.RTROIObservationsSequence = RTROIObservationsSequence

    logging.info("Writing RT Structure file")
    ds.save_as(fname)
    logging.info("File saved.")


def writeDCMheader(filename, dcmStruct):
    logging.info("Setting file meta information...")
    # Populate required values for file meta information
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3' # IOP of RT STORAGE
    file_meta.MediaStorageSOPInstanceUID = dicom.uid.generate_uid()
    file_meta.ImplementationClassUID = "1.3.6.1.4.1.9590.100.1.3.100.9.4"
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.FileMetaInformationGroupLength = 1

    logging.info("Setting dataset values...")
    # Create the FileDataset instance (initially no data elements, but file_meta supplied)
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    if hasattr(dcmStruct, 'PatientName'):
        ds.PatientName = dcmStruct.PatientName

    if hasattr(dcmStruct, 'PatientID'):
        ds.PatientID = dcmStruct.PatientID

    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Set creation date/time
    dt = datetime.datetime.now()
    ds.StructureSetDate = dt.strftime('%Y%m%d')
    ds.StudyDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.StructureSetTime = timeStr
    ds.StudyTime = timeStr

    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = dicom.uid.generate_uid()
    ds.SeriesInstanceUID = dicom.uid.generate_uid()

    if hasattr(dcmStruct, 'StudyInstanceUID'):
        ds.StudyInstanceUID = dcmStruct.StudyInstanceUID

    if hasattr(dcmStruct, 'PatientSex'):
        ds.PatientSex = dcmStruct.PatientSex

    if hasattr(dcmStruct, 'PatientBirthDate'):
        ds.PatientBirthDate = dcmStruct.PatientBirthDate

    if hasattr(dcmStruct, 'StudyID'):
        ds.StudyID = dcmStruct.StudyID

    if hasattr(dcmStruct, 'ReferringPhysicianName'):
        ds.ReferringPhysicianName = dcmStruct.ReferringPhysicianName

    ds.StudyDescription = 'DLROI'
    ds.SeriesDescription = 'DLROI'

    if hasattr(dcmStruct, 'AccessionNumber'):
        ds.AccessionNumber = dcmStruct.AccessionNumber

    if hasattr(dcmStruct, 'InstitutionName'):
        ds.InstitutionName = dcmStruct.InstitutionName

    if hasattr(dcmStruct, 'StationName'):
        ds.StationName = dcmStruct.StationName


    ds.ColorType = 'grayscale'
    ds.Manufacturer = 'CourtLab - MD Anderson'
    ds.Modality = 'RTSTRUCT'
    ds.SoftwareVersion = 'test_v1'
    ds.StructureSetName = 'ROI'
    ds.StructureSetLabel = 'AutoPlan ROI'

    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.Rows = 0
    ds.Columns = 0
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.Width = 0
    ds.Height = 0
    ds.BitDepth = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    return ds


def postprocess(pred, num_volume):
    logging.info("--------------post process----------------")
    volume_count = morphology.label(pred, connectivity=1)
    num_object = len(np.unique(volume_count)) - 1 # subtract the background

    if num_object > num_volume:
        unique_volume = []
        for i in range(num_object+1):
            unique_volume.append(np.count_nonzero(volume_count == i))
            
        unique_volume = sorted(unique_volume)
        pred = morphology.remove_small_objects(pred, np.int(unique_volume[-num_volume-1] - 1))
        logging.info(unique_volume)
        logging.info("removed : {}".format(unique_volume[-num_volume-1] -1))
        
    return pred


def distance(coordA, coordB):
    dist = 0
    for i, coord in enumerate(coordA):
        dist += math.pow(coord - coordB[i],2)
    
    dist = math.sqrt(dist)

    return dist

def initialize_folders(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        os.mkdir(os.path.join(path, 'images'))
        os.mkdir(os.path.join(path, 'labels'))
        os.mkdir(os.path.join(path, 'extras'))
    if not os.path.isdir(os.path.join(path, 'images')):
        os.mkdir(os.path.join(path, 'images'))
    if not os.path.isdir(os.path.join(path, 'labels')):
        os.mkdir(os.path.join(path, 'labels'))
    if not os.path.isdir(os.path.join(path, 'extras')):
        os.mkdir(os.path.join(path, 'extras'))

    return os.path.join(path, 'images'), os.path.join(path, 'labels'), os.path.join(path, 'extras')


def dcm2npy(dicom_path, im_tag='i', rs_tag='i', rois=None):

    patient_tag = os.path.basename(dicom_path)
    logging.info('Running DICOM to Numpy Conversion for patient ' + patient_tag)
    logging.info('***Initializing folder structure')
    #im_path, label_path, extras_path = initialize_folders(destination_path)
    
    logging.info('***Reading DICOM data')
    images, slices, contours, _, _ = read_images_labels(dicom_path, im_tag=None, rs_tag=None, rois=rois)
    # images is a 3D array containing the CT image (z, y, x)
    # slices is a list which contains the dicom info for each DICOM image
    # contours is a dictionary that contains the contour points read from DICOM RTSTRUCT file

    logging.info('***Getting ROI masks')
    labels = get_labels(contours, images.shape, slices, rois=rois)
    # labels will be a 4D array (z, y, x, roi)
    
    dummy_mask = labels[..., 0]
    logging.info('***Saving a dummy mask')
    #RTStruct([dummy_mask], "dummy_roi", slices, fname=os.path.join(destination_path, 'RS_test.dcm'))
    #logging.info('***Done')
    return images, slices, contours, labels, dummy_mask
