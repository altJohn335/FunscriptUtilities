import os
import shutil
import cv2
import coloredlogs
import logging


# Create and start logger object.
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')


def video_type(video):
    """
    Quick comparative test to quickly and easily deduce whether a video being fed in is either VR or 2D.
    This is performed because VR videos are saved as two very similar images side-by-side for each eye,
    if a video is VR we can split it in half and save one-half only. Potentially useful when building
    a ML dataset wherein having two identical images in the same image hinder learning quality.
    :param video: video object passed in.
    :return: true if VR or false if 2D video.
    """

    frames = len(video)  # total frames in video

    video.seek(int(frames / 2))  # jump to half-way of video
    frame = video.next()

    os.makedirs(os.path.join('temp'), exist_ok=True)  # create temp folder

    cv2.imwrite('temp/snapshot.JPG', cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))  # save image

    img = cv2.imread('temp/snapshot.JPG')  # read image
    height, width = img.shape[:2]

    # Cut the image in half
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff]
    s2 = img[:, width_cutoff:]

    # Save each half
    cv2.imwrite('temp/split_1.JPG', cv2.cvtColor(s1, cv2.COLOR_RGB2BGR))
    cv2.imwrite('temp/split_2.JPG', cv2.cvtColor(s2, cv2.COLOR_RGB2BGR))

    # Test if identical
    list_1 = [os.stat('temp/split_1.JPG').st_size, os.stat('temp/split_2.JPG').st_size]  # push both numbers to array
    list_1.sort()  # sort ascending for maths later on

    shutil.rmtree('./temp')  # delete temp folder

    if not list_1[0] / list_1[1] * 100 <= 92:  # if file size between two
        # JPGs differs more than 8% it is likely to be a 2D video.
        logger.debug('VR video detected. Score: ' + str(round(list_1[0] / list_1[1] * 100, 2)) + '%.')
        return True
    else:
        logger.debug('2D video detected. Score: ' + str(round(list_1[0] / list_1[1] * 100, 2)) + '%.')
        return False

