# Haar_Cascade2
Detects multiple shapes. Based off of https://github.com/Senpat/Haar_Cascade 

### Steps
Store frames of video as png files. Use https://www.isimonbrown.co.uk/vlc-export-frames/ Make sure to run vlc as administrator.

Run rotate_and_flip.py to generate all test cases. Triangles are stored trianglepos and squares are stored in squarepos.

Use haar-object-marker to generate description file and manually generate ROI. 

`python haar_positive_creator.py squarepos bgsquare.txt`

Run cropper.py to convert description file to images (only does one crop per line). You need this for negative data.

`python cropper.py inputdescriptionfile outputfolder outputdescriptionfile` or `python cropper.py bgsquare.txt squarecrop squarecrop.txt`

Note that the description file for negative data is formatted differently than positives. cropper.py will create a file called squarecrop that contains a path to all of the files in squarecrop. The idea is you copy the contents of squarecrop.txt to the negative description file of triangle and all the other shapes (but not square).

Run scramble.py (for example `python scramble.py triangleneg.txt` that scrambles all of the lines in the text file.

Run opencv_createsamples to create a positives vector file. 

`opencv_createsamples -info [positive description file] -num [number of data] -w 20 -h 20 -vec [name of positives vector file]`

For example: `opencv_createsamples -info bgsquare.txt -num 450 -w 20 -h 20 -vec positivessquare.vec`

Run opencv_traincascade to train!

`opencv_traincascade -data [data output folder] -vec [positives vector file] -bg [negative description file] -numPos [number of positive images] -numNeg [number of negative images] -numStages [number of stages, usually 10] -w [width] -h [height]`

Command used to train triangles: `opencv_traincascade -data triangledata -vec positivestriangle.vec -bg triangleneg.txt -numPos 1800 -numNeg 1800 -numStages 10 -w 20 -h 20 -miniHitRate 0.5 -maxFalseAlarmRate 0.3`

Notes about training:
  * for numPos and numNeg, use a number slightly smaller that what you actually have because the haar cascade will eat up more data each stage.
  * decrease miniHitRate and maxFalseAlarmRate for more accuracy.

### Files
bgtriangle.txt -> triangle positive description file

bgsquare.txt -> square positive description file. bgsquare2.txt, squarecrop2, and squarecrop2.txt contains MORE data.

negatives -> negative images for all (grass)

squarecrop -> square pictures converted from bgsquare.txt (for negative data for triangle)

trianglecrop -> triangle pictures converted from bgtriangle.txt (for negative data for square)

squarecrop.txt -> text file containing path to all pictures in squarecrop

trianglecrop.txt -> text file containing path to all pictures in trianglecrop 

squareneg.txt -> negative description file for squares - contains allnegatives and trianglecrop.txt

triangleneg.txt -> negative description file for triangles - contains allnegatives squarecrop.txt

squaredata and squaredata2 -> contains .xml files from after training. squaredata3 is from training with bgsquare2.txt included. squaredata3 NOT TRAINED YET.

triangledata -> contains .xml files for triangles with squares in negative data. Used miniHitRate = 0.5, triangledata2 and triangledata3 will contain .xml files from training with miniHitRate = 0.999, mFAR = 0.3, 
