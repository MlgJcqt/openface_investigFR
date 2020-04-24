#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=2)

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
# modelDir = os.path.join(fileDir, '..', 'models')
modelDir = "/root/openface/models"        ### path fixe pour docker sous linux

dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))


def getRep(imgPath):
    if args.verbose:
        print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)
    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        print("Representation:")
        print(rep)
        print("-----\n")
    return rep

# ##### ligne originale : NxN sur toutes les images
# for (img1, img2) in itertools.combinations(args.imgs, 2):               ## extrait 2 images pour faire paire img1 et img2 (args --> parse, voir ligne 53)
#     d = getRep(img1) - getRep(img2)
#     print("Comparing {} with {}.".format(img1, img2))
#     print(
# "  + Squared l2 distance between representations: {:0.3f}".format(np.dot(d, d)))
# #####

## Comparaison 1xN entre 2 dossiers ou 1 dossier 1 fichier + resultats fichiers csv

dirimg1 = args.imgs[0]
dirimg2 = args.imgs[1]
results = args.imgs[2]

fresult = open(results, "w+")
fresult.write("Image 1;Image 2;Score\n")

## lire path 1 commande term : si dossier, lister contenu
if os.path.isdir(dirimg1):
    listimg1 = []                               #liste tout le contenu du dossier correspondant
    for f in os.listdir(dirimg1):
        listimg1.append(dirimg1 + "/" + f )

else :
    listimg1=[dirimg1]

print (os.path.isdir(dirimg2))

## idem path 2
if os.path.isdir(dirimg2):
    listimg2 = []
    for f in os.listdir(dirimg2):
        listimg2.append(dirimg2 + "/" + f )
else :
    listimg2=[dirimg2]

## comparaison chaque image du path 1 versus chaque du path 2
for img1 in listimg1 :
    for img2 in listimg2:
        nom_img1,_ = os.path.splitext(os.path.basename(img1))               #garder juste le nom ("nom_img1") sans le path ni l'extension (",_")
        nom_img2,_ = os.path.splitext(os.path.basename(img2))
        print (nom_img1 + "      vs      " + nom_img2)

        d = getRep(img1) - getRep(img2)
        fresult.write("%s;%s;%.3f\n" % (nom_img1, nom_img2, np.dot(d, d)))

fresult.close()
