#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script removes global rotation in sheet music images and tries to scale
them to a given musical size by looking at the space between staff lines.

This source code form is subject to the terms of the GNU General Public
License v3.0. If a copy of the GPL was not distributed with this file, you
can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
"""

import math
import os
import sqlite3
import sys
import traceback
from multiprocessing import Pool

import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks, pyramid_reduce
from tqdm import tqdm


def normalize(params):
    src_path, args = params

    try:
        # Build destination path
        if args.dst is not None:
            rel_src_path = os.path.relpath(src_path, args.src)
            rel_src_dir = os.path.dirname(rel_src_path)

            src_filename = os.path.basename(rel_src_path)
            basename, _ = os.path.splitext(src_filename)

            dst_dir = os.path.join(args.dst, rel_src_dir)
            dst_path = os.path.join(dst_dir, f"{args.prefix}{basename}.png")
        else:
            dst_path = None

        # Read image
        image = cv2.imread(src_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        src_h, src_w = image_gray.shape[:2]

        # Scale up or down overly tiny or large images
        if src_h > 5500:
            pre_scale = 5000 / src_h
            image_gray = pyramid_reduce(image_gray, 1 / pre_scale, preserve_range=True).astype(np.uint8)
        elif src_h < 1500:
            pre_scale = 2000 / src_h
            image_gray = cv2.resize(image_gray, None, fx=pre_scale, fy=pre_scale, interpolation=cv2.INTER_LINEAR)
        else:
            pre_scale = 1

        # Check if image is already b/w and threshold if not
        is_black_and_white = np.count_nonzero(image_gray == 255) + np.count_nonzero(image_gray == 0) == (image_gray.shape[0] * image_gray.shape[1])
        if not is_black_and_white:
            tresh = 255 - cv2.ximgproc.niBlackThreshold(image_gray, 255, k=0.1, blockSize=51, type=cv2.THRESH_BINARY, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
        else:
            tresh = 255 - image_gray

        # Preprocess image to have better lines
        closed = cv2.morphologyEx(tresh, op=cv2.MORPH_CLOSE, kernel=np.ones((3, 3)))

        # Perform Hough Transform for quasi-horizontal lines on closed image
        tested_angles = np.linspace(np.radians(85), np.radians(95), 1000)
        h, theta, d = hough_line(closed, theta=tested_angles)

        # Detect peaks in Hough Transform
        _, angles, dists = hough_line_peaks(h, theta, d, min_distance=6, threshold=0.65 * h.max())
        angles = np.degrees(angles)

        if len(angles) == 0:
            return src_path, 'Not enough staff lines detected.'

        # Filter out outliers
        valid_idxs = np.where(np.abs(np.mean(angles) - angles) < 0.4)
        if len(valid_idxs[0]) == 0:
            return src_path, 'Not enough staff lines detected.'

        angles = angles[valid_idxs]
        dists = dists[valid_idxs]

        # Sort distances and compute interline differences between adjacent values
        diff = np.diff(sorted(dists))

        if len(diff) < 5 or len(angles) == 0:
            return src_path, 'Not enough staff lines detected.'

        # Compute global rotation angle
        rotation = -90 + np.mean(angles)

        # Get most common interline difference
        hist, bin_edges = np.histogram(diff, bins=50)
        max_bin = np.argmax(hist)

        # Get exact value from histogram data
        vals_in_bin = [val for val in diff if bin_edges[max_bin] <= val <= bin_edges[max_bin + 1]]
        val_mean = np.mean(vals_in_bin)

        # Compute scaling factor from most common interline difference
        scale = pre_scale * ((args.staff_height / 4) / val_mean * math.cos(math.radians(rotation)))

        if args.skip and (scale > 7 or scale < 0.2):
            return src_path, f"Unrealistic scaling factor of {scale}."

        # Compute target image size
        h, w = image.shape[:2]
        dst_w = int(w * scale)
        dst_h = int(h * scale)

        # Do image transformation
        M = cv2.getRotationMatrix2D((0, 0), rotation, 1)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if scale >= 1:
            scaled = cv2.resize(rotated, (dst_w, dst_h), scale, interpolation=cv2.INTER_CUBIC)
        else:
            is_multichannel = len(rotated.shape) > 2
            scaled = pyramid_reduce(rotated, 1 / scale, preserve_range=True, multichannel=is_multichannel).astype(np.uint8)

        # Save target image
        if args.dst is not None:
            os.makedirs(dst_dir, exist_ok=True)
            cv2.imwrite(dst_path, scaled)

        src_resolution = f"{src_w} x {src_h}"
        dst_resolution = f"{dst_w} x {dst_h}"

        return src_path, src_resolution, dst_path, dst_resolution, rotation, scale

    except:
        return src_path, traceback.format_exc()


if __name__ == '__main__':
    import argparse
    import random
    from glob import glob
    from time import strftime

    parser = argparse.ArgumentParser(description='Normalize sheet music images.')
    parser.add_argument('src', type=str, help='path to root directory with source images')
    parser.add_argument('--dst', type=str, default=None, help='path to root directory for result images')
    parser.add_argument('--db', type=str, default=None, help='path to sqlite database for results')
    parser.add_argument('--staff-height', type=float, default=59, help='target pixel height of staves')
    parser.add_argument('--prefix', type=str, default='', help='prefix to be used in result image filenames')
    parser.add_argument('--num', type=int, default=None, help='number of images to sample')
    parser.add_argument('--seed', type=int, default=42, help='seed value for random sampling')
    parser.add_argument('--skip', action='store_true', default=False, help='skip images with unrealistic scaling estimates')

    args = parser.parse_args()

    pool = Pool(os.cpu_count() - 1)
    extensions = ('jpg', 'jpeg', 'png')
    src_paths = []
    for ext in extensions:
        src_paths.extend(glob(f"{args.src}/**/*.{ext}", recursive=True))

    src_paths.sort()

    if args.db is None and args.dst is None and args.use_prefix is None:
        print("No database/result path or prefix given, exiting.", file=sys.stderr)
        exit(1)

    if args.dst is not None:
        os.makedirs(args.dst, exist_ok=True)

    if args.num is not None:
        random.seed(args.seed)
        src_paths = random.sample(src_paths, args.num)
    else:
        args.num = len(src_paths)

    if args.db is not None:
        if os.path.exists(args.db):
            os.remove(args.db)

        datetime = strftime("%Y-%m-%d %H:%M:%S")

        db = sqlite3.connect(args.db)
        cursor = db.cursor()
        cursor.execute('CREATE TABLE images (src_path text, src_resolution text, dst_path text, dst_resolution text, applied_rotation real, applied_scale real)')
        cursor.execute('CREATE TABLE errors (src_path text, error text)')
        cursor.execute('CREATE TABLE config (datetime text, src_root text, dst_root text, prefix text, staff_height real, num_samples integer, seed integer)')
        cursor.execute(f"INSERT INTO config VALUES ('{datetime}','{args.src}','{args.dst}','{args.prefix}',{args.staff_height},{args.num},{args.seed})")
        db.commit()

    params = [(sp, args) for sp in src_paths]

    for results in tqdm(pool.imap_unordered(normalize, params), total=len(params), smoothing=0.1):
        if results is None:
            continue

        if args.db is not None:
            if len(results) == 2:
                src_path, error = results
                error = error.replace("'", "''")
                cursor.execute(f"INSERT INTO errors VALUES ('{src_path}','{error}')")
                db.commit()
                continue
            else:
                src_path, src_resolution, dst_path, dst_resolution, rotation, scale = results
                cursor.execute(f"INSERT INTO images VALUES ('{src_path}','{src_resolution}','{dst_path}','{dst_resolution}',{rotation},{scale})")
                db.commit()

    db.close()

__author__ = "Simon Waloschek"
__copyright__ = "Copyright 2020, Simon Waloschek"
__credits__ = ["Simon Waloschek"]
__license__ = "GNU GPLv3"
__version__ = "1.0.0"
__maintainer__ = "Simon Waloschek"
__email__ = "simon@waloschek.me"
__status__ = "Development"
