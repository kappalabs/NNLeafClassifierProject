import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import shutil


def rotate(in_file, out_file, degrees):
    """
    Rotates given file around its center by given degree, saves it under specified name.

    :param in_file: Location of the input image to be rotated
    :param out_file: Location of the rotated image destination file
    :param degrees: Number of degrees for the image to be rotated by
    :return: Image rotated by given number of degrees around its center
    """
    img = cv2.imread(filename=in_file, flags=0)
    rows_prev, cols_prev = img.shape

    m = cv2.getRotationMatrix2D(center=(cols_prev / 2, rows_prev / 2), angle=degrees, scale=1)

    radians = np.deg2rad(degrees)
    cols_new, rows_new = (abs(np.sin(radians) * rows_prev) + abs(np.cos(radians) * cols_prev),
                          abs(np.sin(radians) * cols_prev) + abs(np.cos(radians) * rows_prev))

    (tx, ty) = ((cols_new - cols_prev) / 2, (rows_new - rows_prev) / 2)
    m[0, 2] += tx
    m[1, 2] += ty

    rotated_img = cv2.warpAffine(src=img, M=m, dsize=(int(cols_new), int(rows_new)))

    cv2.imwrite(out_file, rotated_img)

    return rotated_img


def translate(in_file, out_file, tx, ty):
    """
    Translates given file by specified distance on both X and Y axis.

    :param in_file: Image to be translated
    :param out_file: Output file, where the result will be stored
    :param tx: Distance to translate by on the X axis
    :param ty: Distance to translate by on the Y axis
    :return: Resulting translated image
    """
    img = cv2.imread(filename=in_file, flags=0)
    rows, cols = img.shape

    m = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_img = cv2.warpAffine(src=img, M=m, dsize=(cols, rows))

    cv2.imwrite(out_file, translated_img)

    return translated_img


def mutate(in_dir, filename, out_dir):
    """
    Generates mutations for file in given input directory, outputs them in to specified directory.

    :param in_dir: Directory containing the input image file
    :param filename: Name (only!) of the file containing image to be mutated
    :param out_dir: Directory, where the mutations will be stored
    """
    in_file = join(in_dir, filename)

    # Mutation constants
    rotation_step = 45
    translation_step = 5
    max_translation = 20

    out_file = join(out_dir, filename + "_r")
    for degrees in range(0, 360, rotation_step):
        # Rotations
        rotate(in_file, out_file + str(degrees), degrees)

        # Translations
        rot_file = out_file + str(degrees)
        rot_out_file = out_file + str(degrees) + "_t"
        for translation in range(translation_step, max_translation, translation_step):
            translate(rot_file, rot_out_file + "l" + str(translation), -translation, 0)
            translate(rot_file, rot_out_file + "d" + str(translation), 0, translation)
            translate(rot_file, rot_out_file + "u" + str(translation), 0, -translation)
            translate(rot_file, rot_out_file + "r" + str(translation), translation, 0)

#         TODO: zvětšení/zmenšení, šum


def generate_mutations(in_dir, out_dir):
    """
    Generates mutations of image files in given directory, saves them to specified output directory.

    :param in_dir: Directory containing all images to be mutated.
    :param out_dir: Empty directory where the results will be stored.
    """
    images = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]
    for f in images:
        mutate(in_dir, f, out_dir)
        # return


def normalize(in_dir, out_dir):
    # TODO: normalizace všech obrázků na stejné rozměry pro účely dalšího zpracování
    return


input_directory = 'images'
output_directory = 'out_images'

try:
    shutil.rmtree(output_directory)
except:
    pass
os.makedirs(output_directory)

generate_mutations(input_directory, output_directory)
