# laser_diameter_measure

A tool I made to measure automatically the diameters of cylinders. Here is the setup.
it consist of a webcam (can be replaced by a phone's camera) and line laser that is all.

![full setup](https://github.com/eikonoklastess/laser_diameter_measure/blob/main/photos/PXL_20241008_192854585.jpg)

## Calibration with Checkerboard: find corners and homography

### Checkerboard
![checkerboard](https://github.com/eikonoklastess/laser_diameter_measure/blob/main/photos/Photo%20on%202024-10-08%20at%203.17%E2%80%AFPM.jpg)
### Corners found homogeny calculated
![corners](https://github.com/eikonoklastess/laser_diameter_measure/blob/main/photos/Screenshot%202024-10-08%20at%203.35.05%E2%80%AFPM.png)

## apply homography and grayscale filter

### homogeny applied gray filter applied for better laser detection
![homogeny](https://github.com/eikonoklastess/laser_diameter_measure/blob/main/photos/Screenshot%202024-10-08%20at%203.33.18%E2%80%AFPM.png)

## Calculate cylinder diameter

### place cylinder
![cylinder](https://github.com/eikonoklastess/laser_diameter_measure/blob/main/photos/Photo%20on%202024-10-08%20at%203.18%E2%80%AFPM.jpg)
### cylinder under homogeny and filter
![trace](https://github.com/eikonoklastess/laser_diameter_measure/blob/main/photos/Screenshot%202024-10-08%20at%203.26.14%E2%80%AFPM.png)
### place filtered pixel on a graph which size was determined with the checkerboard's real size
![graph](https://github.com/eikonoklastess/laser_diameter_measure/blob/main/photos/Screenshot%202024-10-08%20at%203.26.31%E2%80%AFPM.png)

the last part of the program is not finished
