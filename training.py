import argparse
import datetime
import os
import sys
from log_util import logging
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from dsets import LunaDataset
from torch.utils.tensorboard import SummaryWriter
from model import LunaModel, UNetWrapper, SegmentationAugmentation
from util import enumerateWithEstimate
import numpy as np
import multiprocessing as mp
import shutil
import hashlib
from dsets_segmentation import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from torch.optim.lr_scheduler import ReduceLROnPlateau

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
#log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


# Used for computeClassificationLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
METRICS_FN_LOSS_NDX = 2
METRICS_ALL_LOSS_NDX = 3
METRICS_PTP_NDX = 4
METRICS_PFN_NDX = 5
METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9
METRICS_SIZE = 10


class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=16,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )

        parser.add_argument('--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action='store_true',
            default=False,
        )

        parser.add_argument('--tb-prefix',
            default='Luna_Runs/seg',
            help="Data prefix to use for Tensorboard run.",
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='none',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.segmentation_model, self.augmentation_model = self.initModel()
        #self.optimizer = self.initOptimizer()
        self.initOptimizer()

    def initModel(self):
        # Seven input channels - 3x3 context slices
        # 1 slice is focus of segmentation
        # Output class indicating if voxel is part of nodule
        # Each downsampling operation adds 1 to depth 
        segmentation_model = UNetWrapper(
            in_channels=7, 
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model

    def initOptimizer(self):
        self.optimizer = Adam(self.segmentation_model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return
        #return Adam(self.segmentation_model.parameters()) # Adam maintains separate LR for each parameter and automatically update LR
        # return SGD(self.segmentation_model.parameters(), lr=0.001, momentum=0.99)


    def initTrainDl(self):
        train_ds = TrainingLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = Luna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=True,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '/_trn_seg_/' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '/_val_seg_/' + self.cli_args.comment)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        self.best_score = 0.0
        self.validation_cadence = 3
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl) # Train for one epoch
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t) # Log metrics after each epoch

            # Only do validation for given cadence, in this case 5, and log images after
            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:

                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t) # Take recall
                if score > self.best_score:
                  self.best_score = max(score, self.best_score) 
                  self.saveModel('seg', epoch_ndx, isBest=True) # Save model and indicate if we are saving as the best model
                else:
                  self.saveModel('seg', epoch_ndx, isBest=False)

                self.logImages(epoch_ndx, 'trn', train_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()
        train_dl.dataset.shuffleSamples()

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()

            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            self.segmentation_model.eval()

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    '''
    Compute normal dice loss for training sample and for only pixels included in label_g
    Multiplying predictions times label gets pseudo-predictions that got every negative pixel right
    Only pixels that will generate loss are false negative pixels 
    Helpful because recall is very important - can't classify a tumor if it isn't detected in the first place
    '''
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                        classificationThreshold=0.5):
        input_t, label_t, series_list, _slice_ndx_list = batch_tup
        
        # Transfer to GPU
        input_g = input_t.to(self.device, non_blocking=True) 
        label_g = label_t.to(self.device, non_blocking=True)

        # Augments as needed if we are in training
        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        # Runs segmentation model
        prediction_g = self.segmentation_model(input_g)

        # Applies fine Dice loss
        diceLoss_g = self.diceLoss(prediction_g, label_g)
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            # Threshold the prediction to get hard Dice but convert to float for multiplication
            predictionBool_g = (prediction_g[:, 0:1]
                                > classificationThreshold).to(torch.float32)
            
            # Compute true positives, false negatives, false positives
            tp = (     predictionBool_g *  label_g).sum(dim=[1,2,3])
            fn = ((1 - predictionBool_g) *  label_g).sum(dim=[1,2,3])
            fp = (     predictionBool_g * (~label_g)).sum(dim=[1,2,3])

            # Store metrics to tensor for future reference - per batch item as opposed to averaged over the batch
            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        '''
        Weighted loss - getting entire population of positive right is 8 times more important than the entire population of negative pixels correct
        Because there is a sacrifice of true negatives to get better recall, can expect large number of false positives - much better to have false positives than false negatives here
        With SGD the push to overpredict would lead to every pixel being returned as positive - Adam can fine tune LR so stressing false negative loss doesn't become overpowering
        '''
        return diceLoss_g.mean() + fnLoss_g.mean() * 8 

    # Dice loss advantage over per pixel cross entropy is handling case where a small portion of overall image is flagged as positive
    # Based fon ratio of correctly segmented pixels to sum of predicted and actual pixels
    # Basically using a per pixel F1 score where population is one images pixels 
    # - because population is entirely contained within one training sample we can use it for training directly
    # - for classification, F1 is not calculable over a single minibatch so we can't use it for training directly
    def diceLoss(self, prediction_g, label_g, epsilon=1):
        diceLabel_g = label_g.sum(dim=[1,2,3]) # Sums over everything except batch dimension to get positively labeled, (softly) pos detected, (softly) correct pos per batch item
        dicePrediction_g = prediction_g.sum(dim=[1,2,3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3])

        # Dice ratio - add one to numerator and denominator to avoid problems if we accidentally have neither preictions nor labels
        diceRatio_g = (2 * diceCorrect_g + epsilon) \
            / (dicePrediction_g + diceLabel_g + epsilon)
 
        return 1 - diceRatio_g # 1 - dice ratio so lower loss is better 


    def logImages(self, epoch_ndx, mode_str, dl):
        self.segmentation_model.eval() # Set model to eval

        images = sorted(dl.dataset.series_list)[:12] # Takes the same 12 CT's by bypassing DL an dusing DS directory, series list might be shuffled - sort
        for series_ndx, series_uid in enumerate(images):
            ct = getCt(series_uid)
            
            # Selects 6 equidistant slices throughout the CT and feed into model, then build an image
            for slice_ndx in range(6):
                ct_ndx = slice_ndx * (ct.hu_a.shape[0] - 1) // 5 
                sample_tup = dl.dataset.getitem_fullSlice(series_uid, ct_ndx)

                ct_t, label_t, series_uid, ct_ndx = sample_tup

                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = pos_g = label_t.to(self.device).unsqueeze(0)

                prediction_g = self.segmentation_model(input_g)[0]
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                label_a = label_g.cpu().numpy()[0][0] > 0.5

                ct_t[:-1,:,:] /= 2000
                ct_t[:-1,:,:] += 0.5

                ctSlice_a = ct_t[dl.dataset.contextSlices_count].numpy()

                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                image_a[:,:,:] = ctSlice_a.reshape((512,512,1)) # CT intensity assigned to all RGB channels for greyscale base image
                image_a[:,:,0] += prediction_a & (1 - label_a)
                image_a[:,:,0] += (1 - prediction_a) & label_a #  FP flagged as red and overlayed on image
                image_a[:,:,1] += ((1 - prediction_a) & label_a) * 0.5 # FN are organe

                image_a[:,:,1] += prediction_a & label_a # TP are green

                # Renormalize data to 0 to 1 range and clamp it, then save to TensorBoard
                image_a *= 0.5
                image_a.clip(0, 1, image_a)

                writer = getattr(self, mode_str + '_writer')
                writer.add_image(
                    f'{mode_str}/{series_ndx}_prediction_{slice_ndx}',
                    image_a,
                    self.totalTrainingSamples_count,
                    dataformats='HWC',
                )

                if epoch_ndx == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:,:,:] = ctSlice_a.reshape((512,512,1))
                    # image_a[:,:,0] += (1 - label_a) & lung_a # Red
                    image_a[:,:,1] += label_a  # Green
                    # image_a[:,:,2] += neg_a  # Blue

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    writer.add_image(
                        '{}/{}_label_{}'.format(
                            mode_str,
                            series_ndx,
                            slice_ndx,
                        ),
                        image_a,
                        self.totalTrainingSamples_count,
                        dataformats='HWC',
                    )
                # This flush prevents TB from getting confused about which
                # data item belongs where.
                writer.flush()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        metrics_dict['percent_all/tp'] = \
            sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = \
            sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = \
            sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100 # Can be larger than 100% because comparing total num pixels labeled as candidate nodules which is tiny fraction of img


        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
            / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                    + "{loss/all:.4f} loss, "
                    + "{pr/precision:.4f} precision, "
                    + "{pr/recall:.4f} recall, "
                    + "{pr/f1_score:.4f} f1 score"
                    ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        log.info(("E{} {:8} "
                    + "{loss/all:.4f} loss, "
                    + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
        ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        writer.flush()

        score = metrics_dict['pr/recall']

        return score


    # Saving only parameters of model - can load them into any model that expects parameters of the same shape even if the class doesn't match the model the params were saved under
    def saveModel(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'luna_data',
            'models',
            self.cli_args.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.cli_args.comment,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module # Gets rid of DataParallel wrapper if it exists

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        if isBest: # Save a second copy if this is the best score we've seen so far
            best_path = os.path.join(
                'luna_data', 'models',
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state')
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())


class LunaTrainingApp:
  def __init__(self, sys_argv=None):
    if sys_argv is None:
      sys_argv = sys.argv[1:] # If caller doesn't provide arguments, get from command line

    parser = argparse.ArgumentParser() # Create a parser for command line arguments
    parser.add_argument('--num-workers',
      help='Number of worker processes for background data loading',
      default=8,
      type=int)

    parser.add_argument('--batch-size',
          help='Batch size to use for training',
          default=32,
          type=int,
      )

    parser.add_argument('--epochs',
        help='Number of epochs to train for',
        default=1,
        type=int,
    )

    parser.add_argument('--balanced',
        help="Balance the training data to half positive, half negative.",
        action='store_true',
        default=False,
    )

    parser.add_argument('--augmented',
      help='Augment the training data',
      action='store_true',
      default=False,
    )

    parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=False,
    )

    parser.add_argument('--augment-offset',
        help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
        action='store_true',
        default=False,
    )

    parser.add_argument('--augment-scale',
        help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
        action='store_true',
        default=False,
    )

    parser.add_argument('--augment-rotate',
        help="Augment the training data by randomly rotating the data around the head-foot axis.",
        action='store_true',
        default=False,
    )

    parser.add_argument('--augment-noise',
        help="Augment the training data by randomly adding noise to the data.",
        action='store_true',
        default=False,
    )

    parser.add_argument('--tb-prefix',
        default='Luna_Runs',
        help="Data prefix to use for Tensorboard run. Using current text chapter",
    )

    parser.add_argument('comment',
        help="Comment suffix for Tensorboard run.",
        nargs='?',
        default='Luna_Project',
    )

    # Parse command line arguments
    self.cli_args = parser.parse_args(sys_argv)

    # Timestamp in the format "Year-Month-Day_Hour.Minute.Second"
    self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

    self.trn_writer = None
    self.val_writer = None
    self.totalTrainingSamples_count = 0

    self.augmentation_dict = {}
    if self.cli_args.augmented or self.cli_args.augment_flip:
        self.augmentation_dict['flip'] = True
    if self.cli_args.augmented or self.cli_args.augment_offset:
        self.augmentation_dict['offset'] = 0.1
    if self.cli_args.augmented or self.cli_args.augment_scale:
        self.augmentation_dict['scale'] = 0.2
    if self.cli_args.augmented or self.cli_args.augment_rotate:
        self.augmentation_dict['rotate'] = True
    if self.cli_args.augmented or self.cli_args.augment_noise:
        self.augmentation_dict['noise'] = 25.0

    self.use_cuda = torch.cuda.is_available()
    self.device = torch.device("cuda" if self.use_cuda else "cpu")

    self.model = self.initModel()
    self.optimizer = self.initOptimizer()

  def initModel(self):
    model = LunaModel()
    if self.use_cuda:
        log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1: # Detects multiple GPU's
            model = nn.DataParallel(model) # Wraps model
        model = model.to(self.device) # Sends model to device
    return model

  def initOptimizer(self):
    return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

  def initTrainDl(self):
    train_ds = LunaDataset(
      val_stride=10, # The stride for calibration in the dataset creation
      isValSet_bool=False, # Not validation set 
      ratio_int = int(self.cli_args.balanced) # True = 1
    )
    batch_size = self.cli_args.batch_size
    if self.use_cuda:
      batch_size *= torch.cuda.device_count() # Adjusts batch size based on the number of available GPUs

    # DL can provide parallel loading
    train_dl = DataLoader(
      train_ds,
      batch_size=batch_size,
      num_workers=self.cli_args.num_workers,
      pin_memory=self.use_cuda, # Pinned memory transfers to GPU quickly
    )

    return train_dl

  def initValDl(self):
    val_ds = LunaDataset(
        val_stride=10, # The stride for calibration in the dataset creation
        isValSet_bool=True, # Validation set
    )

    batch_size = self.cli_args.batch_size
    if self.use_cuda:
        batch_size *= torch.cuda.device_count()

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=self.cli_args.num_workers, # Number of subprocesses for data loading
        pin_memory=self.use_cuda, # DataLoader will copy Tensors into CUDA pinned memory before returning them if true
    )

    return val_dl

  def initTensorboardWriters(self):
    if self.trn_writer is None:
      log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

      self.trn_writer = SummaryWriter(
          log_dir=log_dir + '-trn_cls-' + self.cli_args.comment)
      self.val_writer = SummaryWriter(
          log_dir=log_dir + '-val_cls-' + self.cli_args.comment)


  def main(self):
    mp.set_start_method('spawn', force=True)
    # Logs the start of the application and relevant info
    log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
    train_dl = self.initTrainDl()
    val_dl = self.initValDl()

    #for batch_ndx, batch_tup in enumerate(train_dl):
    #  if batch_ndx % 100 == 0:
    #      log.info("Processed {} batches".format(batch_ndx))


    



    for epoch_ndx in range (1, self.cli_args.epochs + 1):
      log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))
      
      trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
      self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

      valMetrics_t = self.doValidation(epoch_ndx, val_dl)
      self.logMetrics(epoch_ndx, 'val', valMetrics_t)

    if hasattr(self, 'trn_writer'):
      self.trn_writer.close()
      self.val_writer.close()
    
    
  
  def doTraining(self, epoch_ndx, train_dl):
    self.model.train()
    trnMetrics_g = torch.zeros( # Init empty metrics array
        METRICS_SIZE,
        len(train_dl.dataset),
        device=self.device,
    )

    batch_iter = enumerateWithEstimate( # Sets up batch looping with time estimate
        train_dl,
        "E{} Training".format(epoch_ndx),
        start_ndx=train_dl.num_workers,
    )
    for batch_ndx, batch_tup in batch_iter:
        self.optimizer.zero_grad() # Frees any leftover gradient tensors

        loss_var = self.computeBatchLoss( # Computes batch loss
            batch_ndx,
            batch_tup,
            train_dl.batch_size,
            trnMetrics_g
        )

        # Update model weights
        loss_var.backward() 
        self.optimizer.step()

        # # This is for adding the model graph to TensorBoard.
        # if epoch_ndx == 1 and batch_ndx == 0:
        #     with torch.no_grad():
        #         model = LunaModel()
        #         self.trn_writer.add_graph(model, batch_tup[0], verbose=True)
        #         self.trn_writer.close()

    self.totalTrainingSamples_count += len(train_dl.dataset)

    return trnMetrics_g.to('cpu')

  def doValidation(self, epoch_ndx, val_dl):
    with torch.no_grad():
        self.model.eval()
        valMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(val_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            val_dl,
            "E{} Validation ".format(epoch_ndx),
            start_ndx=val_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.computeBatchLoss(
                batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

    return valMetrics_g.to('cpu')


  def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
      input_t, label_t, _series_list, _center_list = batch_tup

      input_g = input_t.to(self.device, non_blocking=True)
      label_g = label_t.to(self.device, non_blocking=True)

      logits_g, probability_g = self.model(input_g)

      loss_func = nn.CrossEntropyLoss(reduction='none')
      loss_g = loss_func(
          logits_g,
          label_g[:,1],
      )
      start_ndx = batch_ndx * batch_size
      end_ndx = start_ndx + label_t.size(0)

      metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = \
          label_g[:,1].detach()
      metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = \
          probability_g[:,1].detach()
      metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = \
          loss_g.detach()

      return loss_g.mean()

  def logMetrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
            ):
    self.initTensorboardWriters()
    log.info("E{} {}".format(
        epoch_ndx,
        type(self).__name__,
    ))

    negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
    negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

    posLabel_mask = ~negLabel_mask
    posPred_mask = ~negPred_mask

    neg_count = int(negLabel_mask.sum())
    pos_count = int(posLabel_mask.sum())

    trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum()) 
    truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())

    falsePos_count = neg_count - neg_correct
    falseNeg_count = pos_count - pos_correct

    

    metrics_dict = {}
    metrics_dict['loss/all'] = \
        metrics_t[METRICS_LOSS_NDX].mean()
    metrics_dict['loss/neg'] = \
        metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
    metrics_dict['loss/pos'] = \
        metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

    metrics_dict['correct/all'] = (pos_correct + neg_correct) \
        / np.float32(metrics_t.shape[1]) * 100
    metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
    metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

    # Double assignment - don't necessarily need separate variables, but will improve later readibility
    precision = metrics_dict['pr/precision'] = truePos_count / np.float32(truePos_count + falsePos_count)
    recall = metrics_dict['pr/recall'] = truePos_count / np.float32(truePos_count + falsePos_count) 

    # Combine precision and recall using F1 score
    metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)

    # Logging

    log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )

    log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
    log.info(
        ("E{} {:8} {loss/neg:.4f} loss, "
              + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
        ).format(
            epoch_ndx,
            mode_str + '_neg',
            neg_correct=neg_correct,
            neg_count=neg_count,
            **metrics_dict,
        )
    )
    log.info(
        ("E{} {:8} {loss/pos:.4f} loss, "
              + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
        ).format(
            epoch_ndx,
            mode_str + '_pos',
            pos_correct=pos_correct,
            pos_count=pos_count,
            **metrics_dict,
        )
    )

    writer = getattr(self, mode_str + '_writer')

    for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

    writer.add_pr_curve(
        'pr',
        metrics_t[METRICS_LABEL_NDX],
        metrics_t[METRICS_PRED_NDX],
        self.totalTrainingSamples_count,
    )

    bins = [x/50.0 for x in range(51)]

    negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
    posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

    if negHist_mask.any():
        writer.add_histogram(
            'is_neg',
            metrics_t[METRICS_PRED_NDX, negHist_mask],
            self.totalTrainingSamples_count,
            bins=bins,
        )
    if posHist_mask.any():
        writer.add_histogram(
            'is_pos',
            metrics_t[METRICS_PRED_NDX, posHist_mask],
            self.totalTrainingSamples_count,
            bins=bins,
        )

if __name__ == '__main__':
  SegmentationTrainingApp().main()