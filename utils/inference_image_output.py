#####################################################################################################################################################################
# xView2                                                                                                                                                            #
# Copyright 2019 Carnegie Mellon University.                                                                                                                        #
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO    #
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY,          # 
# EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, # 
# TRADEMARK, OR COPYRIGHT INFRINGEMENT.                                                                                                                             #
# Released under a MIT (SEI)-style license, please see LICENSE.md or contact permission@sei.cmu.edu for full terms.                                                 #
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use  #
# and distribution.                                                                                                                                                 #
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:                                                         #
# 1. SpaceNet (https://github.com/motokimura/spacenet_building_detection/blob/master/LICENSE) Copyright 2017 Motoki Kimura.                                         #
# DM19-0988                                                                                                                                                         #
#####################################################################################################################################################################


import json
from shapely import wkt
from shapely.geometry import Polygon
import numpy as np 
from cv2 import fillPoly, imwrite
import cv2

def open_json(json_file_path):
    """
    :param json_file_path: path to open inference json file
    :returns: the json data dictionary of localized polygon and their classifications 
    """

    with open(json_file_path) as jf:
        json_data = json.load(jf)
        inference_data = json_data['features']['xy']
        return inference_data

def create_image(inference_data, background_path):
    """
    :params inference_data: json data dictionary of localized polygon and their classifications
    :param background_path: Path to the background image
    :returns: an numpy array of 8-bit grey scale image with polygons filled in according to the key provided
    """

    damage_key = {'un-classified': 1, 'no-damage': 1, 'minor-damage': 2, 'major-damage': 3, 'destroyed': 4}

    mask_img = cv2.imread(background_path)
    
    overlay = mask_img.copy()
    alpha = 0.5
    
    for poly in inference_data:
        damage = poly['properties']['subtype']
        coords = wkt.loads(poly['wkt'])
        poly_np = np.array(coords.exterior.coords, np.int32)
        
        # fillPoly(mask_img, [poly_np], damage_key[damage])
        damage_color = {
            "no-damage":    (0,220,0),      # green
            "minor-damage": (0,255,255),    # yellow
            "major-damage": (0,140,255),    # orange
            "destroyed":    (0,0,255)       # red
        }    
        fillPoly(mask_img,
                [poly_np],
                damage_color[damage])        
    mask_img = cv2.addWeighted(
        overlay,
        alpha,
        mask_img,
        1-alpha,
        0
    )

    return mask_img

def save_image(polygons, output_path):
    """
    :param polygons: np array with filled in polygons from create_image()
    :param output_path: path to save the final output inference image
    """

    # Output the filled in polygons to an image file
    imwrite(output_path, polygons)
  
def create_inference_image(json_input_path, image_output_path, background_path):
    """
    :param json_input_path: Path to output inference json file
    :param image_outut_pat: Path to save the final inference image
    :param background_path: Path to the background image
    """

    # Getting the inference data from the localization and classification 
    inference_data = open_json(json_input_path)

    # Filling in the polygons and readying the image format 
    polygon_array = create_image(inference_data, background_path)

    # Saving the image to the desired location
    save_image(polygon_array, image_output_path)

if __name__ == '__main__': 
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """inference_image_output.py: Takes the inference localization and classification final outputs in json from and outputs an image ready to be scored based off the challenge parameters""")
    parser.add_argument('--input',
                        required=True,
                        metavar='/path/to/final/inference.json',
                        help="Full path to the final inference json")
    parser.add_argument('--output',
                        required=True,
                        metavar='/path/to/inference.png',
                        help="Full path to save the image to")
    parser.add_argument(
        '--background',
        required=True,
        help='post disaster image'
    )
    args = parser.parse_args()

    # Creating the scoring image
    create_inference_image(args.input, args.output, args.background)
