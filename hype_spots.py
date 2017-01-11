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

from scipy import optimize
from skimage import io
import subprocess
from shlex import split

def stkshow(images, fname='/Users/robertf/Downloads/tmp-stk.tif'):
    with TiffWriter(fname, bigtiff=False, imagej=True) as t:
        if len(images.shape)>2:
            for i in range(images.shape[2]):
                t.save(img_as_uint(images[:,:,i]))
        else:
            t.save(img_as_uint(images))
    java_cmd = ["java", "-Xmx5120m", "-jar", "/Users/robertf/ImageJ/ImageJ.app/Contents/Resources/Java/ij.jar"]
    image_j_args = ["-ijpath", "/Applications", fname]
    subprocess.Popen(java_cmd+image_j_args, shell=False,stdin=None,stdout=None,stderr=None,close_fds=True)

class Metadata():
    def __init__(self, pth):
        self.image_table = pandas.read_csv(join(pth, 'Metadata.csv'))
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
        if 'Zindex' in kwargs:
            image_subset_table = image_subset_table[image_subset_table['Zindex'].isin(kwargs['Zindex'])]
        # Group images and sort them then extract filenames of sorted images
        image_groups = image_subset_table.groupby(groupby)
        image_groups = image_groups.apply(lambda x: x.sort_values(sortby))
        fnames_output = {}
        for posname in image_groups[groupby].unique():
            fnames_output[posname] = image_groups.ix[posname]['filename']
        return self._read_images(fnames_output)
    
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


import pandas as pd
class HyPE_Spots():
    def __init__(self, fovs):
        if not isinstance(fovs, dict):
            raise ValueError("Only input of class dict(name:imgs) supported.")
        self.fovs = fovs
    def __getitem__(self, name):
        if isinstance(name, string):
            return self.fovs[name]
        elif isinstance(name, list):
            return (self.fovs[i] for i in name)
    def filter_spots(self, inplace=False, filt_hot_pixels=True):
        if not inplace:
            fovs = {}
        for position, imgs in self.fovs.items():
            if filt_hot_pixels:
                filtered_images = self._remove_hot_pixels(imgs)
            filtered_images = self._background_subtraction(filtered_images)
            filtered_images = self._local_blur(filtered_images)
            if inplace:
                self.fovs[position] = filtered_images
            else:
                fovs[position] = filtered_images
        if not inplace:
            return fovs
    def _background_subtraction(self, images, kernel=(2.5, 2.5, 0), remove_hot_pixels=True):
        images_bg = filters.gaussian(images, kernel)
        return images - images_bg
    def _local_blur(self, images, kernel=(1, 1, 0), intensity_normalize=False):
        if intensity_normalize:
            for i in range(images.shape[2]):
#                 average_bg = np.percentile(images[:,:,i].flatten(), 80)
                images[:,:,i] = images[:,:,i]
        return filters.gaussian(images, kernel)
    def _remove_hot_pixels(self, imgs, 
                           hot_pixels_file='/Users/robertf/Google Drive/Repo/pyimages/hot_pixels_master_array_100_thresh.csv'):
        pixel_idx_df = pd.read_csv(hot_pixels_file)
        for ix, row in pixel_idx_df.iterrows():
            y, x = row.y, row.x
            imgs[y, x, :] = 0
        return imgs
    def find_spot_candidates(self, images, max_peaks=20000):
        max_num_peaks = max_peaks
        peaks = feature.peak_local_max(images, min_distance=1, num_peaks=max_peaks, indices=True)
        peak_spectras = np.zeros((peaks.shape[0], images.shape[2]))
        for i in range(peaks.shape[0]):
            y = peaks[i][0]
            x = peaks[i][1]
            maxies = [np.amax(images[y-1:y+1, x-1:x+1, i]) for i in range(images.shape[2])]
#             maxie = np.amax(images[y-1:y+1, x-1:x+1, :], axis=2)
            peak_spectras[i, :] = maxies
        return peaks, peak_spectras
#     def plot_clusters()
    def register_image_stack(self, images, bead_coords, pad_with=0):
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
        xvec = np.linspace(0, 8, 8)
        yvec = np.linspace(0, 8, 8)
        X,Y = np.meshgrid(xvec,yvec)
        initial_guess = [4, 4, 2] #xo,yo,sigma
        spot = stk[570-4:570+4, 535-4:535+4, 0]
        popt, pcov = optimize.curve_fit(gauss2dFunc, (X, Y), spot.ravel(), p0=initial_guess)
        window_size = 20
        offset_counter = [Counter() for i in range(images.shape[2])]
        global_position = [defaultdict(list) for i in range(images.shape[2])]
        center_masses = [[] for i in range(images.shape[2])]
        for ix in beads:
            y, x = ix[0], ix[1]
            sub_image_stk = images[y-window_size:y+window_size, x-window_size:x+window_size, :]
            for i in range(sub_image_stk.shape[2]):
                sub_img = sub_image_stk[:,:, i]
                neighbors = feature.peak_local_max(sub_img, min_distance=1, num_peaks=2, threshold_abs=350./2**16)
                offsets = neighbors
                for off in offsets:
                    com = ndimage.center_of_mass(sub_img[off[0]-4:off[1]+4, off[0]-4:off[1]+4])
                    com = off-window_size
                    center_masses[i].append(com)
                    off = off-window_size
                    offset_counter[i][tuple(off)] += 1
                    global_position[i][tuple(off)].append((off[0]+ix[0], off[1]+ix[1]))
        t_vec = [np.multiply(i.most_common(1)[0][0], 1) for i in offset_counter]
        reg_stk = np.zeros(images.shape)
        for i, shift in enumerate(t_vec):
            tform = tf.SimilarityTransform(translation=(shift[1], shift[0])) # x,y input expected
            offset_image = tf.warp(images[:,:,i], tform, )
            reg_stk[:,:,i] = offset_image
        return reg_stk, offset_counter, global_position, center_masses

