from __future__ import print_function

import logging
import os

import six

import SimpleITK as sitk
import torch

import radiomics
from radiomics import featureextractor, getFeatureClasses
import numpy as np

settings = {}
settings['correctMask'] = True
settings['preCrop'] = True
settings['minimumROIDimensions'] = 1
settings['label'] = 255

# Get the PyRadiomics logger (default log-level = INFO)
logger = radiomics.logger
logger.setLevel(logging.ERROR)  # set level to DEBUG to include debug log messages in log file

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.addProvenance(False)

def extract_radiomic_features(images, masks):
	(b, c, h, w) = images.shape

	radiomic_features = torch.FloatTensor()
	for i in range(b):
		img, mask = images[i], masks[i]
		img = img.reshape((w, h, c))
		mask = mask.reshape((w, h, 1))
		img = sitk.GetImageFromArray(img)
		mask = sitk.GetImageFromArray(mask)
		featureVector = extractor.execute(img, mask)
		filter_featureVector = torch.from_numpy(np.array(list(featureVector.values())))
		radiomic_features = torch.cat((radiomic_features, filter_featureVector.reshape((1, -1))), 0)
	# radiomic_features -= torch.min(radiomic_features ,dim=0)[0]
	radiomic_features[torch.isnan(radiomic_features)] = 0
	# radiomic_features /= torch.max(radiomic_features, dim=0)[0]
	# radiomic_features[torch.isnan(radiomic_features)] = 1
	return radiomic_features