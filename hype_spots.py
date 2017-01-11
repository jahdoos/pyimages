from IPython import get_ipython

get_ipython().magic('pylab inline')


import pandas
from os.path import join
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import mpld3
mpld3.enable_notebook()

from skimage import filters, exposure, img_as_float, feature, img_as_ubyte, img_as_uint
from sklearn.cluster import KMeans

from scipy import signal



from tifffile import TiffWriter


from skimage import io
import subprocess
from shlex import split

class Metadata():
    def __init__(self, pth):
        self.image_table = pandas.read_csv(join(pth, 'Metadata.csv'))
        fnames = [name.replace("\\", "/") for name in self.image_table.filename.values]
        self.image_table['filename'] = fnames
    def stkread(self, groupby='Position', sortby='TimestampFrame', **kwargs):
        # Input coercing
        for key, value in kwargs.items():
            if not isinstance(value, list):
                kwargs[key] = [value]
        image_subset_table = self.image_table
        # Filter images according to some criteria
        if 'Position' in kwargs:
            image_subset_table = image_subset_table[image_subset_table['Position'].isin(kwargs['Position'])]
        if 'Channel' in kwargs:
            image_subset_table = image_subset_table[image_subset_table['Channel'].isin(kwargs['Channel'])]
        if 'acq' in kwargs:
            image_subset_table = image_subset_table[image_subset_table['acq'].isin(kwargs['acq'])]
        # Group images and sort them then extract filenames of sorted images
        image_groups = image_subset_table.groupby(groupby)
        image_groups = image_groups.apply(lambda x: x.sort_values(sortby))
        fnames_output = {}
        for posname in image_groups['Position'].unique():
            fnames_output[posname] = image_groups.ix[posname]['filename']
        return self._read_images(fnames_output)
    
    def stkshow(self, images):
        fname = self.save_images(images)
        java_cmd = ["java", "-Xmx5120m", "-jar", "/Users/robertf/ImageJ/ImageJ.app/Contents/Resources/Java/ij.jar"]
        image_j_args = ["-ijpath", "/Applications", fname]
        subprocess.Popen(java_cmd+image_j_args, shell=False,stdin=None,stdout=None,stderr=None,close_fds=True)
#         get_ipython().system('{system_call}')
    
    def save_images(self, images, fname = '/Users/robertf/Downloads/tmp_stk.tif'):
        with TiffWriter(fname, bigtiff=False, imagej=True) as t:
            if len(images.shape)>2:
                for i in range(images.shape[2]):
                    t.save(img_as_uint(images[:,:,i]))
            else:
                t.save(img_as_uint(images))
        return fname
    def _read_images(self, filename_dict):
        images_dict = {}
        for key, value in filename_dict.items():
            images_dict[key] = []
            for fname in value:
                img = Image.open(fname)
                images_dict[key].append(img_as_float(np.array(img)))
            images_dict[key] = np.array(images_dict[key]).swapaxes(0, 2)
            images_dict[key] = images_dict[key].swapaxes(0,1)
        return images_dict


class HyPE_Spots():
    def __init__(self, fovs):
        self.fovs = fovs
    def filter_spots(self):
        for key, value in self.fovs.items():
            filtered_images = self._background_subtraction(value)
            filtered_images = self._local_blur(filtered_images)
            self.fovs[key] = filtered_images
    def _background_subtraction(self, images, kernel=(2.2, 2.2, 0)):
        images_bg = filters.gaussian(images, kernel)
        return images - images_bg
    def _local_blur(self, images, kernel=(1.1, 1.1, 0), intensity_normalize=False):
        if intensity_normalize:
            for i in range(images.shape[2]):
#                 average_bg = np.percentile(images[:,:,i].flatten(), 80)
                images[:,:,i] = images[:,:,i]
        return filters.gaussian(images, kernel)
    def find_spot_candidates(self, images, max_peaks=20000):
        max_num_peaks = 200*10**3
        peaks = feature.peak_local_max(images, min_distance=1, num_peaks=max_peaks, indices=True)
        peak_spectras = np.zeros((peaks.shape[0], images.shape[2]))
        for i in range(peaks.shape[0]):
            y = peaks[i][0]
            x = peaks[i][1]
            maxies = [np.amax(images[y-1:y+1, x-1:x+1, i]) for i in range(images.shape[2])]
#             maxie = np.amax(images[y-1:y+1, x-1:x+1, :], axis=2)
            peak_spectras[i, :] = maxies
        return peaks, peak_spectras

    def register_image_stack(self, yxz, images, pad_with=0):
        """
        Translate images to align them.
        
        Parameters
        ----------
        yxz : numpy.array
        images : numpy.array
        
        Returns
        -------
        reg_stack : numpy.array
        """
        assert(images.shape[2]==yxz.shape[0])
        for i in range(images.shape[2]):
            y_shift = yxz[i, 0]
            x_shift = yxz[i, 1] 
            z_shift = yxz[i, 2]
            if y_shift==0 and x_shift==0:
                continue
            ysz, xsz, lambda_sz = images.shape
            current_image = np.zeros(images[:,:, i].shape)
#             current_image = np.pad(current_image, ((), (), ()), mode='edge')
            if y_shift < 0:
                y_get_idx_start = np.abs(y_shift)
                y_get_idx_end = ysz
                y_set_idx_start = 0
                y_set_idx_end = ysz+y_shift
            else:
                y_get_idx_start = 0
                y_get_idx_end = ysz-y_shift
                y_set_idx_start = np.abs(y_shift)
                y_set_idx_end = ysz
            if x_shift < 0:
                x_get_idx_start = np.abs(x_shift)
                x_get_idx_end = xsz
                x_set_idx_start = 0
                x_set_idx_end = xsz+x_shift
            else:
                x_get_idx_start = 0
                x_get_idx_end = xsz-x_shift
                x_set_idx_start = np.abs(x_shift)
                x_set_idx_end = xsz
            current_image[y_set_idx_start:y_set_idx_end, x_set_idx_start:x_set_idx_end] = images[y_get_idx_start:y_get_idx_end, x_get_idx_start:x_get_idx_end, i]
            images[:, :, i] = current_image
        return images

yx_shift = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [-1, -2, 0],
    [-1, -2, 0],
    [-1, -2, 0],
    [-1, -2, 0],
    [-1, -2, 0],
    [0, -2, 0],
    [0, -2, 0],
    [0, -2, 0],
    [0, -2, 0],
])
