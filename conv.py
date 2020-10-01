import numpy as np

class Conv:

    def __init__(self,num_filters,filters,random):
        #Set the number of filters
        self.num_filters = num_filters

        #Set the numpy array of filters
        if random == False:
            self.filters = filters
        else:
            self.filters = np.random.randn(num_filters,3,3)/9

    #Convolution Operation
    def convolution1(self, image, filter1):
        #feature map with dimensions of image <minus 1> for outer edge
        feature_map = np.zeros((image.shape))

        #size of filter being used
        filter_size = filter1.shape

        #loop for each location of the filter in the convolution operation
        for a in range(0,image.shape[0]-filter_size[0]):
            for b in range(0,image.shape[1]-filter_size[0]):
                #identify the part of the image to be convoluted
                select = image[a:a+filter_size[0],b:b+filter_size[0]]

                #sum of the products of each element between image and filter
                #NOTE: IMAGES THAT WERE IN COLOR CAUSE PROBLEMS WITH THE FOLLOWING LINE
                #LFWCROP IMAGES WORK FINE
                feature_map[a][b]=np.sum(select*filter1)

        #return the updated feature map for this convolution
        return feature_map

    #oversees convolution layer (convolution happens in conv1)
    def forward(self, input):
        self.last_input = input
        result = np.zeros((input.shape[0]-2,input.shape[1]-2,self.num_filters))
        
        for reg, i, j in self.iterate(input):
            result[i,j] = np.sum(reg*self.filters,axis=(1,2))

            '''MIGHT NOT NEED THIS/CONVOLUTION1 B/C OF ITERATE AND NP.SUM
            result[i,j] = self.convolution1(input,self.filters)'''
        
        

        #return feature map
        return result

    def iterate(self, img):
        y = img.shape[0]
        x = img.shape[1]

        for b in range(0,y-2):
            for a in range(0,x-2):
                reg = img[b:(b+3),a:(a+3)]
                yield reg, b, a

    def backprop(self, dLoss_dOut,learn):
        dLoss_dFiltersAlt = np.zeros(self.filters.shape)
        dLoss_dFilters = np.zeros((self.filters.shape[1],self.filters.shape[2],self.filters.shape[0]))

        for reg, y, x in self.iterate(self.last_input):
            for fil in range(0,self.num_filters):
                dLoss_dFilters[:,:,fil] += dLoss_dOut[y,x,fil]*reg
                dLoss_dFiltersAlt[fil] += dLoss_dOut[y,x,fil]*reg

        #print(self.filters.shape)

        self.filters-=learn*dLoss_dFiltersAlt
        
        #print("Number of NONZERO Values in Filters: ", np.count_nonzero(self.filters))

        dLastInput = np.zeros((self.last_input.shape[0],self.last_input.shape[1],self.filters.shape[2]))
        dFilters = np.zeros((self.filters.shape[1],self.filters.shape[2],self.filters.shape[0]))

        dLoss_dInput = np.zeros((self.last_input.shape[0],self.last_input.shape[1],self.filters.shape[0]))

        filter_dim = self.filters.shape[1]
        dInput_y = dLoss_dOut.shape[0]
        dInput_x = dLoss_dOut.shape[1]

        #print("dLastInput ->",dLastInput.shape)
        #print("dLoss_dOut ->",dLoss_dOut.shape)

        for y2 in range(0,dInput_y):
            for x2 in range(0,dInput_x):
                for f2 in range(0,self.num_filters):
                    dLoss_dInput[y2:y2+filter_dim,x2:x2+filter_dim,f2] += self.filters[f2] * dLoss_dOut[y2,x2,f2]
                    dFilters[:,:,f2] += self.last_input[y2:y2+filter_dim,x2:x2+filter_dim] * dLoss_dOut[y2,x2,f2]

        #print("dLoss_dFilters ->", dLoss_dFilters.shape)
        #print("dFilters       ->", dFilters.shape)
        #print("dLastInput     ->", dLastInput.shape)
        #print("dLoss_dInput   ->", dLoss_dInput.shape)
        

        return dLoss_dInput