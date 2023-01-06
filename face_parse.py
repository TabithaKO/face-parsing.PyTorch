import cv2
import numpy as np
import pandas as pd 
import os
import boto3
from PIL import Image
import io
import test
import argparse


def prep_avg(scale = None):
  df = pd.read_csv("res/test-res/parts.csv")
  mask = df.to_numpy()
  out = my_skin_detect(mask,1)

  return out

def prep_avg(scale = None):
  df = pd.read_csv("res/test-res/parts.csv")
  mask = df.to_numpy()
  out = my_skin_detect(mask,1)

  return out


def get_mask(img, out):
  # add a row of zeros at the top
  adder = np.zeros((1,np.shape(img)[1]), dtype= np.uint8)
  arr = np.r_[out,adder]
  masked = cv2.bitwise_and(img,img,mask=arr)
  # cv2_imshow(masked)
  # show the half mask
  print("-"*100)
  masked = masked[:masked.shape[0]//2,:,:]
  # cv2_imshow(masked)
  return masked

def avg_channels(channel):
  counter = 0
  sum = 0
  for t in channel:
    for tt in t:
      if tt != 0:
        counter +=1
        sum += tt

  if counter == 0:
     return 0
     
  return (sum/counter)

def get_avg(img):
  results = []
  for t in range(0,3):
    chann = img[:,:,t]
    res = avg_channels(chann)
    results.append(res)
    
  return results

def parse(ref):
  # get the mask of the reference image
  res = prep_avg("half")

  # apply mask
  # print("ref dimensions:",ref.shape)
  # print("res dimensions:",res.shape)
  res_mask = get_mask(ref, res)

  # get the average of the colored pix
  avgs = get_avg(res_mask)

  # check if the averages are 0
  val = sum(np.asarray(avgs))
  # print("sum avgs:",val )

  if val == 0:
    return "non"

  return avgs

def my_skin_detect(src,label):
    dst = np.zeros(np.shape(src), dtype= np.uint8)
    mask = np.logical_and.reduce((src[:,:]==label,))
    dst[mask] = 255
    return dst

def pil_to_cv2(img):
    pil_image = img.convert('RGB') 
    open_cv_image = np.array(pil_image) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return open_cv_image

def main(bucket_name, S3_ID, S3_Key, input_dir, output_name):
    import numpy
    width = 256
    height = 256
    dim = (width, height)

    # My images were in an S3 bucket, be sure to place your correct 
    # S3 bucket key and ID in the strings below. Yes, these are strings.
    AWS_ACCESS_KEY_ID = S3_ID
    AWS_SECRET_ACCESS_KEY = S3_Key

    s3 = boto3.resource('s3',
                        aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    def image_from_s3(key):
      bucket = s3.Bucket(bucket_name)
      image = bucket.Object(key)
      img_data = image.get().get('Body').read()
      return Image.open(io.BytesIO(img_data))

    counter = 0
    output_values = []
    filenames = []

    if bucket_name == "No Bucket":
        files = os.listdir(input_dir)
        filenames = [input_dir+"/" for i in range(0,len(files))]
    else:
        my_bucket = s3.Bucket(bucket_name)
        lis = list(my_bucket.objects.all())
        filenames = [lis[i].key for i in range(0,len(lis))]


    def evaluate(img):
        this_img_avg = parse(img)

        if this_img_avg == "non":
            print("Error! Reference is non")
        else:
            return this_img_avg
        
    for i in range(0,len(filenames)):
      try:
        img = None
        if bucket_name == "No Bucket":
            img = cv2.imread(filenames[i])
        else:
            image_from_s3(filenames[i])
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite("holder/test.jpg",img)

        test.evaluate(dspth="holder/", cp='79999_iter.pth')
        # print("hapa")
        out_val = evaluate(img)
        print("out val is:",out_val)

        if out_val != None:
            output_values.append([filenames[i],out_val])

        counter += 1
        print("------------",counter,"---------------")

      except Exception as e:
          # exc_type, exc_value, exc_traceback = sys.exc_info()
          # traceback.print_exception(exc_value)
          print(f"We have failed!{e}")

      if counter%50 == 0:

        df = pd.DataFrame(output_values)
        df.to_csv(output_name, header=False, index=False)
          


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', type=str, default="No Bucket",
                        help='Name of your S3 bucket')
    parser.add_argument('--S3_ID', type=str, default="No ID",
                        help='Your S3 bucket ID')             
    parser.add_argument('--S3_Key', type=str, default="No Key",
                        help='Your S3 bucket Key')
    parser.add_argument('--input_dir', default="data/input",
                        help='Directory with your original images')
    parser.add_argument('--output_name', type=str, default="data/face_parse_output.csv",
                        help='CSV file to save the BGR metrics of the datasets')

    args = parser.parse_args()
    main(args.bucket_name, args.S3_ID, args.S3_Key, args.input_dir, args.output_name)
