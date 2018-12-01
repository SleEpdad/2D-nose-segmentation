import SimpleITK as sitk
import nibabel as nib
import os
import numpy as np

image =sitk.ReadImage('./train/2/Nose2_boundingboxcut_boundingboxcut.mhd')
image = sitk.GetArrayFromImage(image)
image = image.transpose((2,1,0))
affine = np.diag([1, 1, 1])
image=image[:,:,179]
array_img = nib.Nifti1Image(image,affine)
nib.save(array_img,'nose2_1.nii')

#image_dcm=sitk.ReadImage('/Users/fankun/Data/鼻CTデータ_北村先生_20161029/鼻２/S0000001/ST000001/IM000217')
#image_dcm=sitk.GetArrayFromImage(image_dcm)
#image_dcm=image_dcm.transpose(2,1,0)
#array_img_dcm=nib.Nifti1Image(image_dcm,affine)
# sitk.WriteImage(array_img_dcm,'nose2_217.nii')

#img=nib.load('nose2_217.nii')
#img_arr=img.get_fdata()
#img_arr=np.arange(512*512).reshape(512,512,1)
#image_nii =sitk.ReadImage('nose2_217.nii')
#image_nii = sitk.GetArrayFromImage(image_nii)
#image_nii = image_nii.transpose((2,1,0))

#if(image_dcm[2][2]==img_arr[2][1]):
#    print("right")
#else:
#    print("eroor")




