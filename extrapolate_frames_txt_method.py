import json
import math
import os
import shutil
import sys
import coloredlogs
import logging

# Create and start logger object.
from decord import VideoReader, cpu

from utils import pairwise

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')


def extrapolate_frames(video_path):
    """
    Extrapolate actions from a funscript file. Generates a point at every frame
    by looking at the distance between two points. Uses decord's VideoReader.
    :param video_path: path of the video
    :return: count of images saved
    """


    video_dir, video_filename = os.path.split(video_path)

    assert os.path.exists(video_path)  # assert the video file exists
    try:
        assert os.path.exists(os.path.splitext(video_path)[0] + '.funscript')  # assert the associated
        logger.debug('Successfully found funscript file.')
        # funscript file exists
    except AssertionError:
        logger.error('Funscript file not found. Please make sure the funscript file'
                     ' has the same name and is in the same place as the video.')
        sys.exit(1)

    # Load the VideoReader
    # note: GPU decoding requires decord to be built from source. Uses NVIDIA codecs.
    # See https://github.com/dmlc/decord#install-via-pip. NVIDIA GPUs only.
    decoder = cpu(0)  # can set to cpu or gpu .. decoder = gpu(0)
    video = VideoReader(video_path, ctx=decoder)

    if str(decoder).split('(', 1)[0] == 'cpu':
        logger.warning('GPU processing disabled. To use your GPU for faster processing visit:'
                       ' https://github.com/dmlc/decord#install-via-pip. NVIDIA GPUs only.')

    fpms = video.get_avg_fps() / 1000  # frames per millisecond

    # Load the funscript file
    with open(os.path.splitext(video_path)[0] + '.funscript') as f:
        data = json.load(f)

    actions = data['actions']  # point to the array we require from the funscript JSON array
    print(actions)

    with open('data.txt', 'r+') as outfile:
        outfile.truncate(0)  # need '0' when using r+
        outfile.write('{"actions":[')
        for a, b in pairwise(actions[:len(actions)]):
            # print(a, b)
            distance = math.floor(((b['at'] - a['at']) * fpms))  # how many frames/points can fit between two,
            # print(distance)
            # round down to prevent overshoot.
            for i in range(1, distance + 1):
                formula = ((a['pos']) + i * ((b['pos'] - a['pos']) / (distance + 1)))  # formula for finding the 'pos' key
                # for the points we wish to fit in.
                outfile.write(str({'at': round(((a['at'] * fpms) + i) / fpms), 'pos': round(formula)}))
                outfile.write(',')
                outfile.write('\n')

                actions.append({'at': round(((a['at'] * fpms) + i) / fpms), 'pos': round(formula)})

        outfile.seek(0, 2)  # seek to end of file; f.seek(0, os.SEEK_END) is legal
        outfile.seek(outfile.tell() - 3, 0)  # seek to the second last char of file; f.seek(f.tell()-2, os.SEEK_SET) is legal
        outfile.truncate()
        outfile.write('],')
        del data['actions']  # drop source funscript actions array
        json.dump(data, outfile, indent=4)
        outfile.close()

        with open('data.txt') as f:
            newText = f.read().replace('],{', '],').replace("'", '"')

        with open('data.txt', "w") as f:
            f.write(newText)

        f.close()

    # Load the funscript file
    with open('data.txt', 'r+') as fileX:
        data = json.load(fileX)
        actionsx = data['actions']  # point to the array we require from the funscript JSON array
        actionsx.sort(key=lambda x: x['at'])
        del data['actions']
        json.dump(({'actions': actions}, data), fileX, indent=4)

    shutil.copyfile('data.txt', video_dir + '/' + os.path.splitext(video_filename)[0] + '_extrapolated.funscript')

    # with open(video_dir + '/' + os.path.splitext(video_filename)[0] + '_extrapolated.funscript', 'w') as fp:
    #     json.dump(({'actions': actions}, data), fp, indent=4)  # append new actions array to new JSON and export it.
