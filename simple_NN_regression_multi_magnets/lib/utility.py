import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
"""
Contains functions related for plotting
"""

def create_grid_canvas():
    '''
    Create grid image for plotting
    '''
    # Create an empty canvas (black image)
    img_size = (119*4+50, 119*4+140, 3) # height, width
    canvas = np.zeros(img_size, dtype=np.uint8)

    # Draw grid lines
    grid_color = (100, 100, 100)  # Color of the grid lines (gray)
    grid_spacing = 7*4  # 7mm between points, scaled 4 for visualization
    thickness = -1
    radius = 5

    for i in range(20, 119*4+40, grid_spacing):
        cv2.line(canvas, (i, 20), (i, 119*4+20), grid_color, 1)

    for j in range(20, 119*4+40, grid_spacing):
        cv2.line(canvas, (20, j), (119*4+20, j), grid_color, 1)
    
    #for i in range(20, 400, grid_spacing):
    #    for j in range(20, 400, grid_spacing):
    #        cv2.circle(canvas,(i,j), radius, grid_color, thickness)
    return canvas

def plot_cv2grey(data):
    NUM_SENSORS_X = 4
    NUM_SENSORS_Y = 4
    IMG_SCALING_FACTOR = 100
    IMG_SIZE = (NUM_SENSORS_X*IMG_SCALING_FACTOR, NUM_SENSORS_Y*IMG_SCALING_FACTOR)
    x_array = []
    y_array = []
    z_array = []
    for item in data:
        x,y,z = item
        x_array.append(abs(x))
        y_array.append(abs(y))
        z_array.append(abs(z))
            
    x_img = np.array(x_array).reshape(4,4)
    y_img = np.array(y_array).reshape(4,4)
    z_img = np.array(z_array).reshape(4,4)
    x_maxval = max(400,np.max(x_img)) 
    y_maxval = max(400,np.max(y_img)) 
    z_maxval = max(400,np.max(z_img)) 
    x_img = cv2.resize((255*x_img/x_maxval).astype('uint8'), IMG_SIZE, interpolation = cv2.INTER_NEAREST)
    y_img = cv2.resize((255*y_img/y_maxval).astype('uint8'), IMG_SIZE, interpolation = cv2.INTER_NEAREST)
    z_img = cv2.resize((255*z_img/z_maxval).astype('uint8'), IMG_SIZE, interpolation = cv2.INTER_NEAREST)
    x_img = cv2.flip(x_img, 0)
    y_img = cv2.flip(y_img, 0)
    z_img = cv2.flip(z_img, 0)
    
    cv2.imshow("Image X", x_img)
    cv2.imshow("Image Y", y_img)
    cv2.imshow("Image Z", z_img)
    
    cv2.waitKey(1)
    #edges = cv2.Canny(gray,70,110,apertureSize = 3)
    #lines = cv2.HoughLines(edges,1,np.pi/180,50)


def plot_cv2localization_30_multi_magnets_regression(result, grid_canvas):
    '''
    Cv2 plotting function for sensor separation of 30mm
    Y-axis also flipped for cv2 coordinate system

    Input: 
        result: list of coordinates in meters (m), format [x1, y1, x2, y2, ...]
        grid_canvas: cv2 image, directly retrieved from create_grid_canvas()
    '''
    coor = []

    # Convert and scale coordinates
    for i in range(len(result)//3): ## / -> float, // -> int
        x, y = result[i*3:i*3+2] * 1000  # m to mm
        x, y = x * 4 + 81.28, y * 4 + 43.6  # Scale to fit the plotting box
        coor.append((x, y))

    canvas = np.copy(grid_canvas)

    # Draw points and text on the canvas
    color = (255, 255, 255)  # White color
    thickness = -1  # Filled circle
    radius = 5  # Radius of the point
    centers = []
    texts = []

    i=0
    for x, y in coor:
        center = (int(x), 119*4+40 - int(y))  # Flip y-axis for cv2 coordinate system
        centers.append(center)
        text = f'{i}: ({result[coor.index((x,y))*3]*1000:.2f}, {result[coor.index((x,y))*3+1]*1000:.2f})'
        texts.append((text, (center[0] + 10, center[1] - 10)))
        cv2.circle(canvas, center, radius, color, thickness)
        i=i+1

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)  # White color

    for text, position in texts:
        cv2.putText(canvas, text, position, font, font_scale, text_color, font_thickness)

    # Draw a vertical line for the z-axis
    line_x = 119*4+70  # x-position of the vertical line
    cv2.line(canvas, (line_x, 20), (line_x, 119*4), (255, 255, 255), 1)

    z_texts = []
    # Plot a point on the vertical line to represent the z-axis value
    for i in range(len(result)//3):
        z_value = result[i*3+2] * 1000 * 4 + 20  # Scale to fit the plotting box
        z_center = (line_x, 119*4+20 - int(z_value))
        cv2.circle(canvas, z_center, radius, (255, 255, 255), thickness)  # Red color for z-axis point

        # Add text with z-coordinate
        z_text = f'{i}: {result[i*3+2] * 1000:.2f}'
        z_texts.append((z_text, (line_x + 10, z_center[1] - 10)))  # Adjust the position as needed
    
    for text, position in z_texts:
        cv2.putText(canvas, text, position, font, font_scale, (255, 255, 255), font_thickness)

    # Display the canvas
    cv2.imshow("Point Visualization", canvas)
    cv2.waitKey(1)