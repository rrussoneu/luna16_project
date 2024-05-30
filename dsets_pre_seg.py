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


raw_cache = getCache('part2ch10_raw')

# Holds info for each nodule
'''
  isNodule - what we are training to classify
  diameter - differentiate nodule sizes
  series - to locate ct scan
  center - to find nodule in larger ct image
'''
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

# cache because parsing can be slow
# onDisk - only want to use luna series ID's that are present on disk
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
  
  mhd_list = glob.glob('luna_data/subset*/*.mhd')
  presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
  
  diameter_dict = {}   # keep track of diameter information for cross referencing with annotations
  with open('luna_data/annotations.csv', "r") as f:
    for row in list(csv.reader(f))[1:]:
      series_uid = row[0]
      annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
      annotationDiameter_mm = float(row[4])

      diameter_dict.setdefault(series_uid, []).append((annotationCenter_xyz, annotationDiameter_mm))

  # build candidate list with info in csv file
  candidateInfo_list = []
  with open('luna_data/candidates.csv', "r") as f:
    for row in list(csv.reader(f))[1:]:
      series_uid = row[0]
      if series_uid not in presentOnDisk_set and requireOnDisk_bool:
        # skip if this is a subset not on disk
        continue
      
      isNodule_bool = bool(int(row[4]))
      candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

      candidateDiameter_mm = 0.0
      for annotation_tup in diameter_dict.get(series_uid, []):
        annotationCenter_xyz, annotationDiameter_mm = annotation_tup
        for i in range(3):
          delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
          # radius = diameter / 2, radius / 2 to require that two nodule center 
          # points are not too far apart relative to size
          # bounding box, not distance check
          # if close enough, they are treated as the same nodule
          if delta_mm > annotationDiameter_mm / 4: 
            break
          else:
            candidateDiameter_mm = annotationDiameter_mm
            break
      
      candidateInfo_list.append(CandidateInfoTuple(
        isNodule_bool,
        candidateDiameter_mm,
        series_uid,
        candidateCenter_xyz,
      ))
  candidateInfo_list.sort(reverse=True) # largest nodules samples to non nodules
  return candidateInfo_list

# Loading CT Scans

class Ct:
    def __init__(self, series_uid):
      # doesn't matter which subset a given id is in
        mhd_path = glob.glob('luna_data/subset*/{}.mhd'.format(series_uid))[0]


        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32) # 3 spatial dimensions in array
        ct_a.clip(-1000, 1000, ct_a) # anything outside patient view is considered air an discarded
        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3) # convert directions to array and reshape into into 3x3 matrix

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
      return ct_chunk, center_irc


# Caching
@functools.lru_cache(1, typed=True)
def getCt(series_uid):
  return Ct(series_uid)

# Cache the getCT return value and return values
# After cache is populated getCT won't need to be called again
@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

'''
  5 types of data augmentation:
    - Mirror up-down, left-right, and/or front-back
    - Shift image around by a few voxels
    - Scaling image 
    - Rotating around head-foot axis
    - Adding noise
'''
def getCtAugmentedCandidate(
  augmentation_dict,
  series_uid,
  center_xyz,
  width_irc,
  use_cache=True):

  # Obtain ct_chnk and convert it to a tensor
  if use_cache:
    ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)
  else:
    ct = getCt(series_uid)
    ct_chunk, center_irc = Ct.getRawCandidate(center_xyz, width_irc)
  
  ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

  transform_t = torch.eye(4)

  for i in range(3):

    # When mirroring keep pixel values the same, and just change the orientation
    # No strong correlation between tumor growth and left-right or front-back
    # So we can fip them without changing nature of sample 
    # The index-axis/ Z/ top and bottom of tumor might have a difference but 
    # We assume this is fine

    if 'flip' in augmentation_dict:
      if random.random() > 0.5:
        # grid_sample maps range [-1, 1] to extensors of old and new tensors, and we need to multiply
        # Relevant element of transformation matrix by -1 
        transform_t[i,i] *= -1

    # Shifting nodule candidate shouldn't make a big difference because convolutions
    # Are translation independent and this will make model more robust to imperfectly centered nodules
    # Offset might not be interger number of voxels - data will be resampled using trilinear interpolation 
    # To reduce slight blurring
    # Offset param is max offset expressed in same scale as [-1, 1] range grid_sample expects
    if 'offset' in augmentation_dict:
      offset_float = augmentation_dict['offset']
      random_float = (random.random() * 2 - 1)
      transform_t[i,3] = offset_float * random_float
    
    if 'scale' in augmentation_dict:
      scale_float = augmentation_dict['scale']
      random_float = (random.random() * 2 - 1)
      transform_t[i,i] *= scale_float * random_float

  # X and Y axes have uniform spacing along rows/cols but Z voxels are non-cubic
  # Resampling so Z is the same as along X and Y would result in data being blurry and smeared
  # Instead confine rotations to the X-Y plane
  if 'rotate' in augmentation_dict:
    angle_rad = random.random() * math.pi * 2
    s = math.sin(angle_rad)
    c = math.cos(angle_rad)

    rotation_t = torch.tensor([
      [c, -s, 0, 0],
      [s, c, 0, 0],
      [0, 0, 1, 0 ],
      [0, 0, 0, 1]
    ])

    transform_t @= rotation_t

  affine_t = F.affine_grid(
    transform_t[:3].unsqueeze(0).to(torch.float32),
    ct_t.size(),
    align_corners=False,
  )

  
  augmented_chunk = F.grid_sample(
    ct_t,
    affine_t,
    padding_mode='border',
    align_corners=False,
  ).to('cpu')

  if 'noise' in augmentation_dict:
    noise_t = torch.randn_like(augmented_chunk)
    noise_t *= augmentation_dict['noise']
    augmented_chunk += noise_t
  
  return augmented_chunk[0], center_irc




class LunaDataset(Dataset):
  '''
  val_stride - Set how often to designate sample as member of validation set
  isValSet_bool - Keep training, validation, or everything
  series_uid - If truthy series_uid passed in, instance will only have nodules from that series - for visualization and debugging purposes
  ratio_int - Will control the label for Nth sample and keep track of samples separated by label
  '''
  def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None, sortby_str='random', ratio_int = 0):
    self.ratio_int = ratio_int
    #if candidateInfo
    self.candidateInfo_list = copy.copy(getCandidateInfoList()) # copy return value so cached copy won't be impaceted by altering self.candidateInfo_list

    # partition subset into validation and training
    if series_uid:
      self.candidateInfo_list = [
        x for x in self.candidateInfo_list if x.series_uid == series_uid
      ]

    if isValSet_bool:
      assert val_stride > 0, val_stride
      self.candidateInfo_list = self.candidateInfo_list[::val_stride]
      assert self.candidateInfo_list
    
    elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

    if sortby_str == 'random':
        random.shuffle(self.candidateInfo_list)
    elif sortby_str == 'series_uid':
        self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
    elif sortby_str == 'label_and_size':
        pass
    else:
        raise Exception("Unknown sort: " + repr(sortby_str))

    self.negative_list = [
        nt for nt in self.candidateInfo_list if not nt.isNodule_bool
    ]
    self.positive_list = [
        nt for nt in self.candidateInfo_list if nt.isNodule_bool
    ]

    log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
        self,
        len(self.candidateInfo_list),
        "validation" if isValSet_bool else "training",
        len(self.negative_list),
        len(self.positive_list),
        '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
    ))

    


  def __len__(self):
    if self.ratio_int: # Speed up epochs by harcoding length
      return 200000
    else:
      return len(self.candidateInfo_list)


  # Call this at top of each epoch to randomize the order of samples presented
  def shuffleSamples(self):
    if self.ratio_int:
      random.shuffle(self.negative_list)
      random.shuffle(self.positive_list)


  def __getstate__(self):
    state = self.__dict__.copy()
    # Remove non-picklable entries if there are any
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    # Restore any necessary data or state here

  def __getitem__(self, ndx):
    if self.ratio_int: # ratio_int of 0 means use native balance
      pos_ndx = ndx // (self.ratio_int + 1)

      if ndx % (self.ratio_int + 1): # A non-zero remaineder means this should be negative sample
        neg_ndx = ndx - 1 - pos_ndx
        neg_ndx %= len(self.negative_list) # Overflow = wraparound
        candidateInfo_tup = self.negative_list[neg_ndx]
      else:
        pos_ndx %= len(self.positive_list) # Overflow = wraparound
        candidateInfo_tup = self.positive_list[pos_ndx]
    else:
      candidateInfo_tup = self.candidateInfo_list[ndx] # Returns Nth sample if not balancing

    width_irc = (32, 48, 48) # depth, height, width

    candidate_a, center_irc = getCtRawCandidate(
      candidateInfo_tup.series_uid,
      candidateInfo_tup.center_xyz,
      width_irc,
    ) # candidate_a has shape (32,48,48), depth, height, width

    # manipulate into proper data types and required array dimensions
    candidate_t = torch.from_numpy(candidate_a)
    candidate_t = candidate_t.to(torch.float32)
    candidate_t = candidate_t.unsqueeze(0) # add channel dimension
    pos_t = torch.tensor([ # build classification tensor
      not candidateInfo_tup.isNodule_bool,
      candidateInfo_tup.isNodule_bool
      ],
      dtype=torch.long,
    )
    # return training sample
    return (
      candidate_t,
      pos_t,
      candidateInfo_tup.series_uid,
      torch.tensor(center_irc),
    )

