import SimpleITK as sitk

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames('/Users/fankun/Data/鼻CTデータ_北村先生_20161029/鼻_101/ST000001')
reader.SetFileNames(dicom_names)
image2 = reader.Execute()
image_array = sitk.GetArrayFromImage(image2)  # z, y, x
origin = image2.GetOrigin()  # x, y, z
spacing = image2.GetSpacing()  # x, y, z
sitk.WriteImage(image2, 'test.nii')  # 这里可以直接换成image2 这样就保存了原来的数据成了nii格式了。