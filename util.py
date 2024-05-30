from collections import namedtuple
import numpy as np
import gzip
import datetime
import time
from log_util import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

#from diskcache import FanoutCache, Disk

# positive x is patient left, positive y is posterior, positive z is superior - LPS
IrcTuple = namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = namedtuple('XyzTyple', ['x','y','z'])

'''
Need to flip coordinates from IRC to CRI to align with XYZ
Scale indeces with voxel size
matrix multiply with directions matrix
add offset for the origin
'''
def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1] # reverse order for np array
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a # scale, multiply, add offset
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    #print("Shapes:", coord_a.shape, origin_a.shape, vxSize_a.shape, direction_a.shape)
    # Calculate inverse coordinate transformations
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    #print("Intermediate coordinates:", cri_a)  # Debug print
    
    # Round to nearest whole number
    cri_a = np.round(cri_a)
    #print("Rounded coordinates:", cri_a)  # Debug print

    #irc_z = int(cri_a[2])
    #irc_y = int(cri_a[1])
    #irc_x = int(cri_a[0])
    #print("Integers:", irc_z, irc_y, irc_x)
    
    # Convert to integers and create a tuple
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))

    '''

    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a # inverse of above function steps
    cri_a = np.round(cri_a) # round before converting to ints
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0])) # shuffles and converts to ints
  '''


def importstr(module_str, from_=None):

  if from_ is None and ':' in module_str:
      module_str, from_ = module_str.rsplit(':')

  module = __import__(module_str)
  for sub_str in module_str.split('.')[1:]:
      module = getattr(module, sub_str)

  if from_:
      try:
          return getattr(module, from_)
      except:
          raise ImportError('{}.{}'.format(module_str, from_))
  return module

def enumerateWithEstimate(
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
  ):
  if iter_len is None:
        iter_len = len(iter)

  if backoff is None:
      backoff = 2
      while backoff ** 7 < iter_len:
          backoff *= 2

  assert backoff >= 2
  while print_ndx < start_ndx * backoff:
      print_ndx *= backoff

  log.warning("{} ----/{}, starting".format(
      desc_str,
      iter_len,
  ))
  start_ts = time.time()
  for (current_ndx, item) in enumerate(iter):
      yield (current_ndx, item)
      if current_ndx == print_ndx:
          duration_sec = ((time.time() - start_ts)
                          / (current_ndx - start_ndx + 1)
                          * (iter_len-start_ndx)
                          )

          done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
          done_td = datetime.timedelta(seconds=duration_sec)

          log.info("{} {:-4}/{}, done at {}, {}".format(
              desc_str,
              current_ndx,
              iter_len,
              str(done_dt).rsplit('.', 1)[0],
              str(done_td).rsplit('.', 1)[0],
          ))

          print_ndx *= backoff

      if current_ndx + 1 == start_ndx:
          start_ts = time.time()

  log.warning("{} ----/{}, done at {}".format(
      desc_str,
      iter_len,
      str(datetime.datetime.now()).rsplit('.', 1)[0],
  ))
