import numpy as np

class Pool:

    def __init__(self):
        pass

    def iterate(self, img):
        #y = img.shape[0]
        #x = img.shape[1]
        y, x, _ = img.shape
        y2 = y // 2
        x2 = x // 2

        for b in range(0,y2):
            for a in range(0,x2):
                reg = img[(b*2):(b*2+2),(a*2):(a*2+2)]
                yield reg, b, a

    #pooling operation (renamed "forward" so use is pool.forward)
    def forward(self,input):
        self.last_input = input

        height,width,filters=input.shape
        result = np.zeros((height//2,width//2,filters))

        for reg, i, j in self.iterate(input):
            result[i,j] = np.amax(reg,axis=(0,1))

        return result

    def backprop(self, dL_dOut):
        #Need dL_dInput
        #(math) dL_dInput = dL_dOut * dOut_dT * dT_dInput
        #dL_dOut given as input

        dL_dInput = np.zeros(self.last_input.shape)

        for reg,i,j in self.iterate(self.last_input):
            h,w,f=reg.shape
            npmax = np.amax(reg,axis=(0,1))

        #print(dL_dInput.shape)
        #print(dL_dOut.shape)
        for i2 in range(h):
            for j2 in range(w):
                for f2 in range(f):
                    if reg[i2,j2,f2] == npmax[f2]:
                        dL_dInput[i*2+i2,j*2+j2,f2]=dL_dOut[i,j,f2]
        
        return dL_dInput