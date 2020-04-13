import sys
import cv2
import numpy as np 
from sklearn.mixture import GaussianMixture

def dilationUnknownFgBgNeighbor(unknown_mask, kernal_size, fg_mask, bg_mask):
    """
        Use cv2.diliation() to get  foreground and background mask after dilation 
        args:
            unknown_mask: shape(# of image row, # of image column)
                the mask of the unknown region of the trimap
            kernal_szie: 
                the image filter size used by cv2.dilation
            fg_mask: shape(# of image row, # of image column)
                the mask of the foreground region of the trimap
            bg_mask: shape(# of image row, # of image column)
                the mask of the background region of the trimap
        return:
            dila_fg_mask: shape(# of image row, # of image column)
                foreground mask after dilation
            dila_bg_mask: shape(# of image row, # of image column)
                background mask after dilation
    """
    kernel = np.ones((kernal_size,kernal_size),np.uint8)
    dilation_alpha = cv2.dilate(unknown_mask, kernel, iterations = 1)
    
    dila_fg_mask = np.logical_and(fg_mask, dilation_alpha)
    dila_bg_mask = np.logical_and(bg_mask, dilation_alpha)
    
    return dila_fg_mask, dila_bg_mask

if __name__ == '__main__':
    # read src image and trimap
    file_name = 'gandalf'
    img = cv2.imread('./img/'+ file_name + '.png', cv2.IMREAD_COLOR)
    img = img/255
    trimap = cv2.imread('./trimap/'+ file_name + '.png', cv2.IMREAD_GRAYSCALE) # grayscale - two dimension
    
    # prepare masks by trimap
    fg_mask = trimap == 255
    bg_mask = trimap == 0
    unknown_mask = True ^ np.logical_or(fg_mask, bg_mask)
    # fill in known fg, bg, and alpha
    alpha = np.zeros(trimap.shape)
    alpha[fg_mask] = 1
    alpha[unknown_mask] = np.nan 
    fg = img*np.repeat(fg_mask[:, :, np.newaxis], 3, axis=2) 
    bg = img*np.repeat(bg_mask[:, :, np.newaxis], 3, axis=2)
    
    
    # get coordinates of unknown pixels
    unknown = np.argwhere(np.isnan(alpha))

    

    # for showing how many pixels were computed
    PASS = 0
    total = len(unknown)

    # Calculate each unknown points to get F、B、Alpha
    while len(unknown) > 0:
        y,x = unknown[0]
        #print("Progress:{}, y = {}, x = {}, img = {}".format(PASS + 1, y, x, img[y][x]))
        
        #########################################
        # TODO: prepare data points and apply GMM


        # Dilation parameters used for calling cv2.dilation()
        initKernal_size = 80
        iterIncreKernal = 80
        # Perpare data points -> the surrounding pixels of the unknown part 
        # loop and find until the fg and bg points are enough to apply GMM
        kernal_size = initKernal_size
        while True: 
            temp_alpha = np.zeros(trimap.shape)
            temp_alpha[y, x] = 1 # assgin only one unknown points to mask for dilation
            dila_fg_mask, dila_bg_mask = dilationUnknownFgBgNeighbor(temp_alpha, kernal_size, fg_mask, bg_mask) # get fg mask, bg mask after dilation process 
            
            if (np.count_nonzero(dila_fg_mask) >= 30 and np.count_nonzero(dila_bg_mask) >= 30):
                print("kernal_size={}\n".format(kernal_size))
                break
            kernal_size += iterIncreKernal # if not enough, increase the filter size for dilation process
        dila_fg = img * np.repeat(dila_fg_mask[:, :, np.newaxis], 3, axis=2)
        dila_bg = img * np.repeat(dila_bg_mask[:, :, np.newaxis], 3, axis=2)
        dila_fg = dila_fg.reshape(dila_fg.shape[0]*dila_fg.shape[1], 3)
        dila_bg = dila_bg.reshape(dila_bg.shape[0]*dila_bg.shape[1], 3)
        fg_reduced = dila_fg[~np.all(dila_fg == 0, axis=1)] # exclude the datapoints they aren't the surrounding pixels of the unknown part 
        bg_reduced = dila_bg[~np.all(dila_bg == 0, axis=1)] 
            

        # Apply GMM on fg and bg points
        num_fg_components = 3
        num_bg_components = 3
        gmm_max_iter = 100
        gmm_n_init = 5
        fg_gmm = GaussianMixture(n_components=num_fg_components, max_iter=gmm_max_iter, n_init=gmm_n_init).fit(fg_reduced)
        bg_gmm = GaussianMixture(n_components=num_bg_components, max_iter=gmm_max_iter, n_init=gmm_n_init).fit(bg_reduced)
        
        # extract the inforamtion of trained GMM to apply optimization
        inv_cov_fgmm = np.linalg.inv(fg_gmm.covariances_)
        inv_cov_bgmm = np.linalg.inv(bg_gmm.covariances_)
        mean_fgmm = fg_gmm.means_
        mean_bgmm = bg_gmm.means_

        #########################################
    
        ##########################################################################
        # TODO: try to set different initial of alpha and optimize F, B, and alpha
        
        C = img[y][x]
        sigmaC = 0.01
        initAlpha = 0.1
        iterNum = 30
        invSigmaSqr = 1 / (sigmaC**2)
        I = np.eye(3)
        maxlike_pair = np.zeros(2)
        max_F = np.zeros(3)
        max_B = np.zeros(3)
        max_a = np.zeros(0)
        maxlike = -np.inf
        minlike = 1e-6

        # Optimization F, G, Alpha to each pair of fg and bg, then get the maximum likelihood
        for i in range(num_fg_components):
            mu_Fi = mean_fgmm[i]
            invSigma_Fi = inv_cov_fgmm[i]
            for j in range(num_bg_components):
                mu_Bj = mean_bgmm[j]
                invSigma_Bj = inv_cov_bgmm[j]

                F = []
                B = []
                a = initAlpha
                lastlike = -np.inf # record the last likelihood value for early stop
                for it in range(iterNum):
                    # Update F and B, fix a,  
                    A11 = invSigma_Fi + I * (a**2) * invSigmaSqr
                    A12 = I * a * (1 - a) * invSigmaSqr
                    A22 = invSigma_Bj + I * ((1 - a) ** 2) *  invSigmaSqr
                    A = np.vstack( (np.hstack((A11, A12)), np.hstack((A12, A22))) )
                    b1 = invSigma_Fi @ mu_Fi + C * a * invSigmaSqr
                    b2 = invSigma_Bj @ mu_Bj + C * (1 - a) * invSigmaSqr
                    b = np.concatenate((b1, b2))
                    X = np.linalg.solve(A, b)
                    
                    F = np.maximum(0, np.minimum(1, X[0:3]))
                    B = np.maximum(0, np.minimum(1, X[3:6]))
                    
                    # Update alpha, fix F, B
                    a = np.maximum(0, np.minimum(1, np.dot((C - B), (F - B)) / (np.linalg.norm(F - B)**2)))                
                    
                    # calculate likelihood
                    L_C = -((np.linalg.norm(C - a * F - (1 - a) * B)**2) * invSigmaSqr)
                    L_F = (- ((F - mu_Fi) @ invSigma_Fi @ (F - mu_Fi)) / 2)
                    L_B = (- ((B - mu_Bj) @ invSigma_Bj @ (B - mu_Bj)) / 2)
                    like = L_C + L_F + L_B
                    
                    if maxlike < like:
                        maxlike = like
                        #maxlike_pair = [i, j] # record i, j index
                        max_F = F
                        max_B = B
                        max_a = a
                        
                    # early stop, if have been converge 
                    if abs(like - lastlike) <= minlike:
                        break
                        
                    lastlike = like
        
        alpha[y][x] = max_a
        fg[y][x] = max_F
        bg[y][x] = max_B
        
        
                
        ##########################################################################
        
        unknown = np.delete(unknown, 0, 0)      
        PASS += 1
        sys.stdout.write("\rprogress:\t{}/{}\n".format(PASS,total))
        sys.stdout.flush()

    
    target_scene = cv2.imread('landscape.png',cv2.IMREAD_COLOR)
    target_scene = cv2.resize(target_scene, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_CUBIC)/255
    
    ####################################################
    # TODO: attach the result foreground to target scene
    
    alpha_extend = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    composite_img = 255 * (alpha_extend * fg + (1 - alpha_extend) * target_scene)
    img_name = 'final/sigmaC={}_a={}_iterNum={}_Fcomp={}_Bcomp={}_kernal={}_iterIncreKernal={}_gmmMaxiter={}_gmmNinit={}_fileName={}.png'.format(sigmaC, initAlpha, iterNum, num_fg_components, num_bg_components, initKernal_size, iterIncreKernal, gmm_max_iter, gmm_n_init, file_name)
    cv2.imwrite(img_name, composite_img)
    
    #grayscale_imgName = 'final/GsigmaC={}_a={}_iterNum={}_Fcomp={}_Bcomp={}_kernal={}_iterIncreKernal={}_gmmMaxiter={}_gmmNinit={}_fileName={}.png'.format(sigmaC, initAlpha, iterNum, num_fg_components, num_bg_components, initKernal_size, iterIncreKernal, gmm_max_iter, gmm_n_init, file_name)
    #cv2.imwrite(grayscale_imgName, alpha * 255)
    
    ####################################################
    
