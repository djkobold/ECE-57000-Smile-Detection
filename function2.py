#!/usr/bin/env python

import numpy as np


#normalize inputs and weights
#check stanford

#preprocessing
def prep(image):
    recolor = False
    resize = False
    reshape = False

    #change this to adjust size of the image
    dim = 64
    new_size = (dim,dim)

    #measurements taken to check if adjustments are needed
    image_rgb = image.convert('RGB')
    width, height = image_rgb.size

    #check if image is in grayscale
    for x in range(width):
        for y in range(height):
            red,green,blue = image_rgb.getpixel((x,y))
            if red != green or red != blue or green != blue:
                recolor = True
                break

    #check if image is a square
    #NOTE: CURRENTLY ASPECT RATIO WILL BE CHANGED IN RESIZE PROCESS
    #MIGHT NEED TO CHANGE IF EXPRESSION CANNOT BE DETECTED WITH
    #ADJUSTED ASPECT RATIOS
    if width != height:
        reshape = True

    #check if image is correct size
    if width != dim or height != dim:
        resize = True

    #recolor image if needed
    if recolor:
        image = image.convert('LA')

    #resize image if needed
    if resize:
        image = image.resize(new_size)
    
    #return the updated image
    return image

#relu operation
def relu(feature_map_list):
    #empty output array of the size of the feature map
    relu_map = np.zeros(feature_map_list.shape)
    
    #loop through each feature map
    for feature_map_num in range(0, len(feature_map_list)):
        #loop through each pixel of the image/feature map/relu map
        for a in range(0,feature_map_list[feature_map_num].shape[0]):
            for b in range(0,feature_map_list[feature_map_num].shape[1]):
                #replace value in relu map with original value in feature map if greater than 0
                relu_map[a][b] = max(0,feature_map_list[feature_map_num][a][b])
    return relu_map

#pooling operation
def pool(relu_map_list,size,stride):
    #empty array for pooling operation, size based on size of pooling mask and the stride length
    #pooling_result = np.zeros(relu_map_list[-1].shape-(size-1))

    #alternate code
    pooling_result = np.zeros((np.uint16((relu_map_list[-1].shape[0]-(size-1))/stride),np.uint16((relu_map_list[-1].shape[1]-(size-1))/stride)))

    pooling_result_list = []

    #iterate through each relu map in the list
    for relu_map_num in range(0,len(relu_map_list)):
        #row of the output matrix
        p_row = 0
        
        #iterate through the relu map
        for a in range(0,relu_map_list[relu_map_num].shape[0]-size-1,stride):
            #column of the output matrix
            p_col = 0
            for b in range(0,relu_map_list[relu_map_num].shape[1]-size-1,stride):
                #place the maximum value in the range in the pooling result matrix
                pooling_result[p_row][p_col]=np.max(relu_map_list[relu_map_num][a:a+size,b:b+size])
                
                #increment the column of the pooling result matrix
                p_col = p_col + 1

            #increment the row of the pooling result matrix
            p_row = p_row + 1
        
        #add the new result to the pooling result list which will later be returned
        pooling_result_list.append(pooling_result)

    return pooling_result_list

def check(filename, directory):
    fn = filename.split('.')[0] + '.jpg'
    sm = open(directory + '\\SMILE_list.txt')
    nsm = open(directory + '\\NON-SMILE_list.txt')

    if fn in sm.read():
        return 1
    elif fn in nsm.read():
        return 0
    else:
        return -1

'''def iterate(img):
    y = img.shape[0]
    x = img.shape[1]
    y2 = int(y/2)
    x2 = int(x/2)

    for b in range(0,y2):
        for a in range(0,x2):
            reg = img[(b*2):(b*2+2),(a*2):(a*2+2)]
            yield reg, b, a'''


'''def conv_bprop(dLoss_dOut,learn,filters,r_input):
    dLoss_dFilters = np.zeros(filters.shape)

    for reg, y, x in iterate(r_input):
        for fil in range(0,len(filters)):
            dLoss_dFilters[fil] += dLoss_dOut[y,x,fil]*reg

    filters-=learn*dLoss_dFilters

    return filters'''
    
