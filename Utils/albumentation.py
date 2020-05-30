from torchvision import transforms
import albumentations as A
import random
import numpy as np

class Albumentation:
  """
  Helper class to create test and train transforms using Albumentations
  """
  def __init__(self, transforms=[]):
    self.transforms = A.Compose(transforms)

  def __call__(self, img):
    img = np.array(img)
    return self.transforms(image=img)['image']





