import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert to rover-centric coordinates
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = np.absolute(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[0]).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    # Apply a rotation
    yaw_rad = np.pi * yaw / 180
    xpix_rotated = xpix*np.cos(yaw_rad) - ypix*np.sin(yaw_rad)
    ypix_rotated = xpix*np.sin(yaw_rad) + ypix*np.cos(yaw_rad)
    # Return the result  
    return xpix_rotated, ypix_rotated

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = np.int_(xpos + (xpix_rot/scale))
    ypix_translated = np.int_(ypos + (ypix_rot/scale))
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def process_img (image, src, dst):
    # converts an input image into the desired system usable images
    # convert image into HSV color space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # locate the drivable path for navigation
    path_threshed = color_thresh(hsv_img, rgb_thresh=(0, 0, 180))
    # loacte the walls and non navigable objects in the image
    #obs_threshed = 1 - color_thresh(image, rgb_thresh=(70, 70, 70))
    # find the target sample rocks in the image
    sample_threshed = color_thresh(hsv_img, rgb_thresh=(0, 117, 117))

    # warpe the images for mapping and object locations
    path_warped = perspect_transform(path_threshed, src, dst)
    #obs_warped = perspect_transform(obs_threshed, src, dst)
    sample_warped = perspect_transform(sample_threshed, src, dst)

    return path_warped, sample_warped #, obs_warped

def mean_angle(array):
    # check the if input array is not empty and if so returns the mean
    # polar to degree conversion of the values or return 0
    if len(array) > 0:
        return np.mean(array * 180 / np.pi)
    else:
        return 0

def mean_dist(array):
    # check the if input array is not empty and if so returns the mean
    # of the values or return 0
    if len(array) > 0:
        return np.mean(array)
    else:
        return 0

def perception_step(Rover):
    # converts the camera images into local and global coordinates.
    # checks images for sample objects and obstacles
    # Perform perception steps to update Rover()

    # check that the rover is not picking up a sample rock or is it a skip
    # frame for faster processing.
    if Rover.picking_up == 0 and Rover.send_pickup is False and Rover.skip_next:
        Rover.skip_next = False # skip the next image
        if Rover.pitch < Rover.pitch_cutoff or Rover.pitch > 360 - Rover.pitch_cutoff:
            img = Rover.img
            # Apply color threshold to identify navigable terrain/obstacles/rock samples
            path_warped, sample_warped = process_img(img, Rover.source, Rover.destination)

            # Update Rover.vision_image (this will be displayed on left side of screen)
            # mask the obstacles to remove camera blind spots
            #Rover.vision_image[:,:,0] = obs_warped * 255
            Rover.vision_image[:,:,1] = sample_warped * 255
            Rover.vision_image[:,:,0] = path_warped * 255

            # Convert map image pixel values to rover-centric coords
            xpix_sample, ypix_sample = rover_coords(sample_warped)
            xpix_path, ypix_path = rover_coords(path_warped)
            #xpix_obs, ypix_obs = rover_coords(obs_warped)

            # Convert rover-centric pixel values to world coordinates
            #x_world_obs, y_world_obs = pix_to_world(xpix_obs, ypix_obs, Rover.pos[0],
            #                                        Rover.pos[1], Rover.yaw,
            #                                        Rover.worldmap.shape[0], Rover.scale)
            x_world_sample, y_world_sample = pix_to_world(xpix_sample, ypix_sample, Rover.pos[0],
                                                      Rover.pos[1], Rover.yaw,
                                                      Rover.worldmap.shape[0], Rover.scale)
            x_world_path, y_world_path = pix_to_world(xpix_path, ypix_path, Rover.pos[0],
                                                      Rover.pos[1], Rover.yaw,
                                                      Rover.worldmap.shape[0], Rover.scale)

            # Update Rover worldmap (to be displayed on right side of screen)
            #Rover.worldmap[y_world_obs, x_world_obs, 0] += 1
            Rover.worldmap[y_world_sample, x_world_sample, 1] += 1
            Rover.worldmap[y_world_path, x_world_path, 2] += 1

            # Convert rover-centric pixel positions to polar coordinates
            # Update Rover pixel distances and angles
            Rover.nav_dists, Rover.nav_angles = to_polar_coords(xpix_path, ypix_path)

            # Check that the path ahead is clear of walss or objects and that the
            #  rover is not too close to any of them.
            nav_mean_angle = mean_angle(Rover.nav_angles) # the paths mean direction
            nav_mean_dist = mean_dist(Rover.nav_dists) # the path length
            Rover.can_go_forward = nav_mean_angle > -1 * Rover.angle_forward and \
                                   nav_mean_angle < Rover.angle_forward and \
                                   nav_mean_dist > Rover.mim_wall_distance

            if sample_warped.any():
                # A rock has been detected so calculate direction to the rock
                Rover.sample_dists, Rover.sample_angles = to_polar_coords(xpix_sample, ypix_sample)
                Rover.sample_detected = True
                Rover.mode = 'sample'
                Rover.turn_dir = 'none'
            elif Rover.can_go_forward:
                # lost the sample and no objects in the path
                Rover.sample_detected = False
                Rover.mode = 'forward'
            else:
                # lost sample and there are objects / wall in the way
                Rover.sample_detected = False
                Rover.mode = 'turn_around'
    #elif Rover.vel == 0 and Rover.pitch < 10 * Rover.pitch_cutoff or Rover.pitch > 360 - Rover.pitch_cutoff * 10:
    #    # has the rover driven up the wall
    #    #Rover.mode = 'backward'
    else:
        Rover.skip_next = True
    return Rover
