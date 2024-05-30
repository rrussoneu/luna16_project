from collections import namedtuple
import os
import glob
import functools
import csv
import SimpleITK as sitk
import numpy as np
from util import XyzTuple, xyz2irc
import torch
from torch.utils.data import Dataset
from disk_util import getCache
import copy
from log_util import logging
import random
import math
import torch.nn.functional as F

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


raw_cache = getCache('part2ch13_raw')

MaskTuple = namedtuple('MaskTuple', 'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask')

# Holds info for each nodule
'''
  isNodule - what we are training to classify
  diameter - differentiate nodule sizes
  series - to locate ct scan
  center - to find nodule in larger ct image
'''
CandidateInfoTuple = namedtuple('CandidateInfoTuple', 'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')

# cache because parsing can be slow
# onDisk - only want to use luna series ID's that are present on disk
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
  
  mhd_list = glob.glob('luna_data/subset*/*.mhd')
  presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
  
  diameter_dict = {}   # keep track of diameter information for cross referencing with annotations
  
  # build candidate list with info in csv file
  candidateInfo_list = []

  with open('luna_data/annotations_with_malignancy.csv', "r") as f:
    for row in list(csv.reader(f))[1:]:
      series_uid = row[0]
      annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
      annotationDiameter_mm = float(row[4])
      isMal_bool = {'False': False, 'True': True}[row[5]]

      candidateInfo_list.append(
          CandidateInfoTuple(
              True,
              True,
              isMal_bool,
              annotationDiameter_mm,
              series_uid,
              annotationCenter_xyz,
          )
      )

  

  with open('luna_data/candidates.csv', "r") as f:
      for row in list(csv.reader(f))[1:]:
          series_uid = row[0]

          if series_uid not in presentOnDisk_set and requireOnDisk_bool:
              continue

          isNodule_bool = bool(int(row[4]))
          candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

          if not isNodule_bool:
              candidateInfo_list.append(
                  CandidateInfoTuple(
                      False,
                      False,
                      False,
                      0.0,
                      series_uid,
                      candidateCenter_xyz,
                  )
              )
  valid_list = []
  num_invalid = 0
  for tup in candidateInfo_list:
    mhd_path = glob.glob('luna_data/subset*/{}.mhd'.format(tup.series_uid))
    if (len(mhd_path) == 0):
      num_invalid += 1
    else:
      valid_list.append(tup)

  log.info("Num invalid: {}".format(num_invalid))

  #candidateInfo_list.sort(reverse=True) # largest nodules samples to non nodules
  valid_list.sort(reverse=True)
  return valid_list

@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    # Take list of candidates for the series UID from the dict
    # Default to a fresh/empty list of we cannot find it
    # Append present candidateInfo_tup to it
    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid,
                                      []).append(candidateInfo_tup)

    return candidateInfo_dict

# Loading CT Scans

class Ct:
    def __init__(self, series_uid):
      # doesn't matter which subset a given id is in
        mhd_path = glob.glob('luna_data/subset*/{}.mhd'.format(series_uid))[0]


        ct_mhd = sitk.ReadImage(mhd_path)
        #ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32) # 3 spatial dimensions in array
        #ct_a.clip(-1000, 1000, ct_a) # anything outside patient view is considered air an discarded
        self.series_uid = series_uid
        #self.hu_a = ct_a
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3) # convert directions to array and reshape into into 3x3 matrix

        candidateInfo_list = getCandidateInfoDict()[self.series_uid]
        self.positiveInfo_list = [
          candidate_tup
          for candidate_tup in candidateInfo_list
          if candidate_tup.isNodule_bool
        ]
        
        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)
        
        # Takes indices of mask slices with nonzero cound and makes into list
        self.positive_indexes = (self.positive_mask.sum(axis=(1,2)).nonzero()[0].tolist())

    def getRawCandidate(self, center_xyz, width_irc):
      center_irc = xyz2irc( 
        center_xyz, 
        self.origin_xyz,
        self.vxSize_xyz,
        self.direction_a,
        )

      slice_list = []
      for axis, center_val in enumerate(center_irc):
        start_ndx = int(round(center_val - width_irc[axis]/2))
        end_ndx = int(start_ndx + width_irc[axis])

        assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis ])

        if start_ndx <= 0:
          # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
          #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
          start_ndx = 0
          end_ndx = int(width_irc[axis])

        if end_ndx > self.hu_a.shape[axis]:
          # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
          #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
          end_ndx = self.hu_a.shape[axis]
          start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

        slice_list.append(slice(start_ndx, end_ndx))
      
      ct_chunk = self.hu_a[tuple(slice_list)]
      pos_chunk = self.positive_mask[tuple(slice_list)]
      return ct_chunk, pos_chunk, center_irc

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu = -700):
      boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool_) # Start with all false tensor same size as ct
    

      for candidateInfo_tup in positiveInfo_list: # Loop over all the nodules
        center_irc = xyz2irc(
          candidateInfo_tup.center_xyz,
          self.origin_xyz,
          self.vxSize_xyz,
          self.direction_a,
        )

        # Get center voxel indices - starting point
        ci = int(center_irc.index)
        cr = int(center_irc.row)
        cc = int(center_irc.col)

        index_radius = 2

        '''
        Find bounding box for when density drops below threshold
        BB should have one voxel border of low desnity tissue at least on one side 
        '''

        try:
          while self.hu_a[ci + index_radius, cr, cc] > threshold_hu \
              and self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
            index_radius += 1
        except IndexError:
          index_radius -= 1
        
        row_radius = 2

        try:
          while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
              self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
            row_radius += 1
        except IndexError:
          row_radius -= 1
        
        col_radius = 2
        try:
          while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
              self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
            col_radius += 1
        except IndexError:
          col_radius -= 1


        # After we get nodule radius mark the bounding box
        boundingBox_a[
          ci - index_radius: ci + index_radius + 1,
          cr - row_radius: cr + row_radius + 1,
          cc - col_radius: cc + col_radius + 1
        ] = True

      # Restrict mask to above density threshold
      mask_a = boundingBox_a & (self.hu_a > threshold_hu)

      return mask_a


# Caching
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
  return Ct(series_uid)

# Cache the getCT return value and return values
# After cache is populated getCT won't need to be called again
@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc


@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes


class Luna2dSegmentationDataset(Dataset):
  def __init__(self,
                val_stride=0,
                isValSet_bool=None,
                series_uid=None,
                contextSlices_count=3,
                fullCt_bool=False, # Use every slice in CT if true for data set - useful for evaluating end to end performance, use False for validation in training because limiting to positive masks
          ):
    self.contextSlices_count = contextSlices_count
    self.fullCt_bool = fullCt_bool

    if series_uid:
        self.series_list = [series_uid]
    else:
        self.series_list = sorted(getCandidateInfoDict().keys())

    # Partition list of series into training and validation sets - entire CT scan with nodule candidates will be in either val or train set
    if isValSet_bool:
        assert val_stride > 0, val_stride
        self.series_list = self.series_list[::val_stride] # Keep only every val_stride'th element starting with 0
        assert self.series_list
    elif val_stride > 0:
        del self.series_list[::val_stride] # If training, delete every val_stride'th element instead
        assert self.series_list

    # Loop over series_UID's we want and get total number of slices and list of ones of interest
    self.sample_list = []
    for series_uid in self.series_list:
        index_count, positive_indexes = getCtSampleSize(series_uid)

        if self.fullCt_bool: # Extentend sample_list with every slice of CT using range
            self.sample_list += [(series_uid, slice_ndx)
                                  for slice_ndx in range(index_count)]
        else:
            self.sample_list += [(series_uid, slice_ndx) # Here take only ones of interest
                                  for slice_ndx in positive_indexes]

    self.candidateInfo_list = getCandidateInfoList() # Cached
    # Filter candidateInfo_list to contain only nodule candidates with series_uid in set of series
    series_set = set(self.series_list) # Set for faster lookup
    self.candidateInfo_list = [cit for cit in self.candidateInfo_list
                                if cit.series_uid in series_set] # Filters out series_uid not in set

    # List for only positive samples to use during training
    self.pos_list = [nt for nt in self.candidateInfo_list
                        if nt.isNodule_bool] # Want only actual nodules for data balancing

    log.info("{!r}: {} {} series, {} slices, {} nodules".format(
        self,
        len(self.series_list),
        {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
        len(self.sample_list),
        len(self.pos_list),
    ))

  def __len__(self):
      return len(self.sample_list)

  # Retreive data in three forms 
  # - full slice of CT given a series_uid and ct_ndx
  # - cropped area around a nodule used for training data
  # - DataLoader needs samples via an integer ndx and dataset needs to return appropraite type based on training vs validation
  # Converts from integer ndx to either full slice or training crop as appropriate
  def __getitem__(self, ndx):
    series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)] # Modulo does wrapping
    return self.getitem_fullSlice(series_uid, slice_ndx)

  def getitem_fullSlice(self, series_uid, slice_ndx):
    ct = getCt(series_uid)
    ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512)) # Preallocate output

    start_ndx = slice_ndx - self.contextSlices_count
    end_ndx = slice_ndx + self.contextSlices_count + 1
    for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
        context_ndx = max(context_ndx, 0) # When we reach beyond bounds of the ct_a, duplicate first or last slice
        context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
        ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

    # HU = 0 g/cc (air, approximately) is -1000 and 1 g/cc (water) is 0.
    # The lower bound - eliminate negative density stuff used to indicate out of POV
    # The upper bound - eliminate any weird hotspots/ clamp down bone
    ct_t.clamp_(-1000, 1000)

    pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0)

    return ct_t, pos_t, ct.series_uid, slice_ndx

'''
Train on 64x64 crops around positive candidates taken randomly from a 96x96 crop centered on the nodule
Also include three slices of context in both directinos and extra channels to 2D segmentation
Makes training more stable and converge more quickly - training on full slices did not work well enough but the 64x64 semirandom crop works well
Whole slice training gives a needle in a haystack situation
'''

class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.ratio_int = 2

  def __len__(self):
      return 300000

  def shuffleSamples(self):
    random.shuffle(self.candidateInfo_list)
    random.shuffle(self.pos_list)

  # Similar to validation set but sample from pos_list and call getItem_trainingCrop with candidate info tuple 
  # since we need series and exact center location not just slice
  def __getitem__(self, ndx):
    candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
    return self.getitem_trainingCrop(candidateInfo_tup) # Similar to getItem_trainingCrop but with a different sized crop passed in and returning another array with a crop of the positive mask

  def getitem_trainingCrop(self, candidateInfo_tup):

    # Limit pos_a to center slice being segmented and construct 64x64 crops of the 96x96 given by getCtRawCandidate
    # Return tuple with the same items as validation dataset

    ct_a, pos_a, center_irc = getCtRawCandidate( # Get candidate with some extra surrounding
        candidateInfo_tup.series_uid,
        candidateInfo_tup.center_xyz,
        (7, 96, 96),
    )
    pos_a = pos_a[3:4] # Take one element slice means keeping third dimension which will be the output channel

    # Crop CT and mask
    row_offset = random.randrange(0,32) 
    col_offset = random.randrange(0,32)
    ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset+64,
                                  col_offset:col_offset+64]).to(torch.float32)
    pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset+64,
                                    col_offset:col_offset+64]).to(torch.long)

    slice_ndx = center_irc.index

    return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx


class PrepcacheLunaDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.candidateInfo_list = getCandidateInfoList()
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_uid)

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        # candidate_t, pos_t, series_uid, center_t = super().__getitem__(ndx)

        candidateInfo_tup = self.candidateInfo_list[ndx]
        getCtRawCandidate(candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz, (7, 96, 96))

        series_uid = candidateInfo_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)

            getCtSampleSize(series_uid)
            # ct = getCt(series_uid)
            # for mask_ndx in ct.positive_indexes:
            #     build2dLungMask(series_uid, mask_ndx)

        return 0, 1 #candidate_t, pos_t, series_uid, center_t