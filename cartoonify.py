#############################################################
# FILE : cartoonify.py
# WRITER : Ilan Vysokovsky, ilan.vys, 207375528
# EXERCISE : intro2cs1 ex5 2021
# DESCRIPTION: A program with multiple functions that are taken
# from the image processsing world.
# the main program will create a cartoon version of an image
# recieved.
#############################################################
import math
import sys

import ex5_helper

def separate_channels(image):
    '''
    Recieves an image with multiple color channels and seperates
    the image into multiple pictures of a single channel.
    :param image: The multi-channel image
    :return: array of single-channel images
    '''
    separate_channels_arr = []
    channels_num = len(image[0][0])
    
    for channel in range(0, channels_num):
        separate_channels_arr.append([])
        
        for row in range(0, len(image)):
            separate_channels_arr[channel].append([])
            
            for col in range(0, len(image[row])):
                separate_channels_arr[channel][row].append(
                    image[row][col][channel])
    
    return separate_channels_arr

def combine_channels(image):
    '''
    Recieves an array of images with single color channels and combines
    the images to a single multi-channel image.
    :param image: array of single-channel images
    :return: multi-channel image
    '''
    combined_channels_arr = []
    channels_num = len(image)
    
    for row in range(0, len(image[0])):
        combined_channels_arr.append([])
        
        for col in range(0, len(image[0][row])):
            combined_channels_arr[row].append([])
            
            for color in range(0, channels_num):
                combined_channels_arr[row][col].append(image[color][row][col])
            
    return combined_channels_arr

def RGB2grayscale(colored_image):
    '''
    Recieves an image with multiple color channels
    and turns the image to grayscale.
    :param image: multi-channel image
    :return: greyscale image
    '''
    garyscale_image = []
    for row in range(0, len(colored_image)):
        garyscale_image.append([])
        for col in range(0, len(colored_image[0])):
            garyscale_image[row].append(round(colored_image[row][col][0] * \
                0.299 + colored_image[row][col][1] * \
                0.587 + colored_image[row][col][2] * 0.114))
    return garyscale_image

def blur_kernel(size):
    '''
    Recieves a size and calculates a blur kernel
    with the size recieved, with values that are an avarege of the
    `size` to the power of `size` 
    :param size: the size of the kernel
    :return: blur kernel
    '''
    kernel = []
    for i in range(0, size):
        kernel.append([])
        for j in range(0, size):
            kernel[i].append(1 / (size * size))
    return kernel            

def calc_sum_of_kernel(row, col, image, kernel):
    '''
    Recieves a row and column, an image and a kernel,
    and calcultes the new value for the image in the index 
    [row][col], using the blur kernel
    :param row: the row index we want to calculate
    :param col: the col index we want to calculate
    :param image: the image to blur image
    :param kernel: the blur kernel
    :return: value of index in new blurred image
    '''
    new_block = get_block(row, col, image , len(kernel))
    sum_kernel = 0
    
    for i in range(0, len(new_block)):
        for j in range(0, len(new_block[0])):
            sum_kernel += (new_block[i][j] * kernel[i][j])
    
    if (sum_kernel < 0):
        return 0
    if (sum_kernel > 255):
        return 255
    
    return round(sum_kernel)
    
def apply_kernel(image, kernel):
    '''
    Recieves an image and a blur kernel,
    and applies the blur in the imageusing the kernel.
    :param image: an image to blur
    :return: blurred image
    '''
    new_image = []
    for row in range(0, len(image)):
        new_image.append([])
        for col in range(0, len(image[0])):
            new_image[row].append(calc_sum_of_kernel(row, col, image, kernel))
    return new_image

def bilinear_interpolation(image, y, x):
    '''
    Recieves an image and a a position,
    and calcultes the color in the index recieved
    :param image: an image to calculate bilinear interpolation
    :param y: y coordinate (row)
    :param x: x coordinate (col)
    :return: value if the bilnear interpolation
    '''
    interpolation_sum = 0
    
    if y > (len(image) - 1):
        y = len(image) - 1
        
    if x > (len(image[0]) - 1):
        x = len(image[0]) - 1
        
    if y < 0:
        y = 0
        
    if x < 0:
        x = 0
        
    floor_x = math.floor(x)
    floor_y = math.floor(y)
    ceil_x = math.ceil(x)
    ceil_y = math.ceil(y)
    
    x -= floor_x
    y -= floor_y

    interpolation_sum += (image[floor_y][floor_x] * (1 - x) * (1 - y))
    interpolation_sum += (image[ceil_y][floor_x] * y * (1 - x))
    interpolation_sum += (image[floor_y][ceil_x] * x * (1 - y))
    interpolation_sum += (image[ceil_y][ceil_x] * x * y)
        
    return round(interpolation_sum)

def resize(image, new_height, new_width):
    '''
    Recieves an image with new height and width,
    and calcultes a new image with the requested height 
    and width.
    :param image: the image to resize
    :param new_height: the size of the new image height
    :param new_width: the size of the new image width
    :return: reszied image
    '''
    row_ratio = (len(image) - 1) /  (new_height - 1)
    col_ratio = (len(image[0]) - 1) / (new_width - 1)
    new_image = []
    
    for i in range(0, new_height):
        new_image.append([])
        for j in range(0, new_width):
            new_image[i].append(bilinear_interpolation(
                image, i * row_ratio, j * col_ratio))
    
    return new_image            

def rotate_90(image, direction):
    '''
    Recieves an image and a direction, and turns 
    the image 90 degrees to desired direction.
    :param image: the image to rotate
    :param direction: the direction to rotate the image to
    :return: rotated image
    '''
    new_image = []
    if direction == 'R':
        for i in range(0, len(image[0])):
            new_image.append([])
            for j in range(len(image) - 1, -1, -1):
                new_image[i].append(image[j][i])
    
    if direction == 'L':
        for i in range(0, len(image[0])):
            new_image.insert(0, [])
            for j in range(len(image) - 1, -1, -1):
                new_image[0].insert(0, image[j][i])
        
    return new_image

def get_block(row, col, image, kernel_size):
    '''
    Creates an array of arrays with recieved height and width 
    and fills the indexes outside the image with the value of
    `image` in the row and column recieved.
    :param row: the row index of the middle of the block to return
    :param col: the col index of the middle of the block to return
    :param image: the image to create the block from
    :param kernel_size: the size of the wanted block
    :return: the kernel with requested size
    '''
    new_kernel = []
    half_kernel_size = math.floor(kernel_size / 2)
    for i in range(int(row - half_kernel_size), 
                   int(row + half_kernel_size + 1)):
        new_kernel.append([])
        for j in range(int(col - half_kernel_size), 
                       int(col + half_kernel_size + 1)):
            if i < 0 or j < 0:
                new_kernel[i + (half_kernel_size - row)] \
                    .append(image[row][col])
                continue
            if i >= len(image) or j >= len(image[row]):
                new_kernel[i + (half_kernel_size - row)]. \
                    append(image[row][col])
                continue
            
            new_kernel[i + (half_kernel_size - row)].append(image[i][j])
    return new_kernel

def calc_threshold(image, c):
    '''
    Calculates the avarege from the values in the image recieved,
    to determine the threshold to return.
    :param image: the image to calculate threshold to
    :param c: the defined size to help set the amount of 
    darkness a pixel should have to be considered an edge
    :return: the threshild of the image recieved
    '''
    image_sum = 0
    for i in image:
        for j in i:
            image_sum += j
            
    threshold = image_sum / (len(image) * len(image))
    r = len(image) // 2
    if threshold - c > image[r][r]:
        return 0
    else:
        return 255
       
def get_edges(image, blur_size, block_size, c):
    '''
    Creates a blurred image with all the edges highlighted by the given params
    :param image: the image to create the edges image from
    :param blur_size: the size of the blur kernel
    :param block_size: the size of the wanted block to check
    :param c: the defined size to help set the amount of 
    darkness a pixel should have to be considered an edge
    :return: an image with highlighted edges
    '''
    blurred_image = apply_kernel(image, blur_kernel(blur_size))
    new_image = []
    for row in range(0, len(image)):
        new_image.append([])
        for col in range(0, len(image[0])):
            block = get_block(row, col, blurred_image, block_size)
            new_image[row].append(calc_threshold(block, c))

    return new_image

def quantize(image, N):
    '''
    Creates an image with `N` colors from the single-channel image recieved
    :param image: the image to quantize
    :param N: the number of desired colors in the new image
    :return: quantized image
    '''
    new_image = []
    for row in range(0, len(image)):
        new_image.append([])
        for col in range(0, len(image[0])):
            new_val = math.floor(image[row][col] * (N / 255)) * (255 / N)
            new_image[row].append(round(new_val))

    return new_image

def quantize_colored_image(image, N):
    '''
    Creates an image with `N` colors from the multi-channel image recieved
    :param image: the image to quantize
    :param N: the number of desired colors in the new image
    :return: quantized image
    '''
    separated_channels_img = separate_channels(image)
    for i in range(0, len(separated_channels_img)):
        separated_channels_img[i] = quantize(separated_channels_img[i], N)
    return combine_channels(separated_channels_img)

def add_mask(image1, image2, mask):
    '''
    Creates an image from the combination of the two images
    recieved, with `mask` affectiong the formula.
    :param image1: the first image to add mask to
    :param image2: the secend image to add mask to
    :param mask: an array of 0's and 1's affectiong the combination formula
    :return: a masked image
    '''
    new_image = []
    if isinstance(image1[0][0], int):
        for i in range(0, len(image1)):
            new_image.append([])
            for j in range(0, len(image1[0])):
                new_image[i].append(round(image1[i][j] * \
                    mask[i][j] + image2[i][j] * (1 - mask[i][j])))
    else:
        separated_channels_img1 = separate_channels(image1)
        separated_channels_img2 = separate_channels(image2)
        for channel in range(0, len(separated_channels_img1)):
            new_image.append([])
            for i in range(0, len(image1)):
                new_image[channel].append([])
                for j in range(0, len(image1[0])):
                    new_image[channel][i].append(
                        round(separated_channels_img1[channel][i][j] * \
                            mask[i][j] + \
                            separated_channels_img2[channel][i][j] * \
                            (1 - mask[i][j])))
        return combine_channels(new_image)
       
    return new_image

def maskify(image): 
    '''
    Creates masked image containing only black and white pixles
    :param image: the image to to manipulate
    :return: a masked image with obly two possible colors
    '''
    new_image = []
    for i in range(0, len(image)):
        new_image.append([])
        for j in range(0, len(image[0])):
            if image[i][j] == 0:
                new_image[i].append(1)
            else:
                new_image[i].append(0)
    return new_image
                
def cartoonify(image, blur_size, th_block_size, th_c, quant_num_shades):
    '''
    Creates cartooned version of an image recieved.
    :param image: the image to to cartoonify
    :param blur_size: the size of the blur kernel
    :param th_block_size: the size of the block used to check edges
    :param c: the param used to help determine the edges of an image
    :param quant_num_shades: the amount of colors the new image will have
    :return: a cartoon image
    '''
    grayscale_image = RGB2grayscale(image)
    edges_image = get_edges(grayscale_image, blur_size, th_block_size, th_c)
    quantized_image = quantize_colored_image(image, quant_num_shades)
    mask_image = maskify(edges_image)
    
    
    separated_channels_img = separate_channels(quantized_image)
    for i in range(0, len(separated_channels_img)):
        separated_channels_img[i] = add_mask(
            edges_image, separated_channels_img[i], mask_image)
            
    
    return combine_channels(separated_channels_img)

if __name__ == "__main__":
    try:
        image_source = sys.argv[1]
        cartoon_dest = sys.argv[2]
        max_im_size = sys.argv[3]
        blur_size = sys.argv[4]
        th_block_size = sys.argv[5]
        th_c = sys.argv[6]
        quant_num_shades = sys.argv[7]

        image = ex5_helper.load_image(image_source)
        cartoon_image = cartoonify(
            image, 
            int(blur_size), 
            int(th_block_size), 
            int(th_c), 
            int(quant_num_shades))
        ex5_helper.save_image(cartoon_image, cartoon_dest)
    
    except IndexError:
        print("args were incorrect. please call `cartoonify` with " \
                "[image_source] [cartoon_dest] [max_im_size] [blur_size] " \
                "[th_block_size] [th_c] [quant_num_shades]")