# Behind the Mask: What Photos on Twitter Reveal

Analysis of mask wearing in Twitter photos

This project aims to utilize a neural network to train and predict whether or not a photo contains a person wearing a mask. There is an ongoing global pandemic in which wearing a face mask (or other covering) reduces the risk and rate of virus transmission, thus saving lives. Our goal is to establish a machine learning model that can both distinguish between mask-wearers and non-mask-wearers to help predict when and where a spike in cases is likely.

# The data science problem

Can mask wearing in photos on Twitter help predict COVID infection rates, giving hospitals and medical workers insight to prepare for spikes.

# The model

This Model was slightly adapted from [Mask Detection using MobilenetV2](https://www.kaggle.com/mirzamujtaba/face-mask-detection). We started with a [Kaggle dataset](https://www.kaggle.com/andrewmvd/face-mask-detection) that contains 853 images of individuals wearing masks and not wearing masks.  This dataset included an annotation file of xmls that contained information about the associated image.  This information was size/shape of the image and where the mask was located on the face.  This was our train/test data.  The model uses BeautifulSoup to read the xml files into a dictionary of the faces.  Then OpenCV reads the images, extracts the faces and preprocesses the data. The image is labeled and saved to an associated list.  Lastly a MobilenetV2 model is used and fine-tuned before presenting results.  MobileNetV2 is a model from Google that was chosen because it is low-power and versatile.  

# Acquiring the data
The Twitter data was acquired through a TwitterCrawler written to scrape images through the Tweepy interface using the hashtags #nomaskselfie and #maskup.  224 images were downloaded along with a CSV containing their text and geo-referenced information, if given.  In this instance, the images were sorted by hand to verify only the ones that contain a human face.

# The process
The training Kaggle images were reshaped into 224,224 then processed through a MobileNetV2 model using Keras.  The initial xml files included with the images contained the location of the masks on people's faces, so no manual bounding boxes were required for this project.  Our model did preprocess the images through such processes as "Flatten","Dropout", "Dense", and "MaxPooling2D".  Once the model was loaded back in using a python script called "RunOnTwitter.py", the Twitter data had to be resized to 224,224.  Then predictions were made on 224 photos.

# Conclusions
Our model predicted the majority of Twitter images were mask wearers (129) vs non mask wearers(95) with an 89% accuracy on 2 runs of 20 epochs each.  Our model stabilized around epoch 12 in the first run, and after there was some slight improvement in performance on the second run of 20 epochs.  Ultimately, our model ended up predicting those without a mask considerably more accurately than those with a mask.  We believe this may be due to the variety of masks that people wear, maybe flowers or some other pattern makes it harder to predict.  

The major issue that arose through building an image classification model involved the images during preprocessing.  It is important to get the images correctly sized and converted back from an array and that can be challenging, especially in the prediction step.  Running the initial training model was computer intensive, especially since training a neural network requires multiple runs.  This model was intended to be used as a product to sell to other companies like hospitals and/or local governments to predict areas of higher infection rates via images. A GIS map shows the potential of mapping those that do contain geo-referenced information.  

# Next Steps

We realize the limited time we have to complete this project and here we will give suggestions as to next steps in case we or others have more time to dedicate to this project.

  1. Find additional data/photo sources with geotags included which will allow us to add depth to the analysis and pin down specific locations and then make inferences about the rates of virus transmission.
  2. Predict dates the US will reach specific infection thresholds.
  3. Fine-tune the model to increase its usefulness and accuracy.
  4. Explore the entire process at a later date with the benefit of knowing how the pandemic plays out and compare that knowledge with the predictions we made at the time of model construction.
  5. Pitch to government agencies (especially hospitals) as a prediction tool for estimating the number of cases to expect and prepare for based on mask-wearing on social media.

# List of files:

- TwitterCrawler.ipynb is used to scrape images from Twitter
- covid_confirmed_usafacts.csv is a daily total of confirmed cases per county obtained [here](https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/)

# Resources

- [Training images](https://www.kaggle.com/andrewmvd/face-mask-detection)
- [Starter notebook](https://www.kaggle.com/andrewmvd/face-mask-detection)
- [Another mask database of images](https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset?)

# Authors

Melissa Anthony,
Tim Rauen,
David Sutton
