import sys
import cv2
import numpy as np 
from sklearn.mixture import GaussianMixture
count = 10000
def dilationUnknownFgBgNeighbor(unknown_mask, kernal_size, fg_mask, bg_mask):
    
    kernal_size = kernal_size
    kernel = np.ones((kernal_size,kernal_size),np.uint8)
    dilation_alpha = cv2.dilate(unknown_mask, kernel, iterations = 1)
    
    dila_fg_mask = np.logical_and(fg_mask, dilation_alpha)
    dila_bg_mask = np.logical_and(bg_mask, dilation_alpha)
        
    global count
#    if count % 1000 == 0:
#        da = "localdilation/da_kernal={}_count={}.png".format(kernal_size, count)
#        df = "localdilation/df_kernal={}_count={}.png".format(kernal_size, count)
#        db = "localdilation/db_kernal={}_count={}.png".format(kernal_size, count)
#        cv2.imwrite(da, dilation_alpha * 255)
#        cv2.imwrite(df, dila_fg_mask * 255)
#        cv2.imwrite(db, dila_bg_mask * 255)
#    count = count - 1
    
    return dila_fg_mask, dila_bg_mask
if __name__ == '__main__':
    # read src image and trimap
    file_name = 'woman'
    img = cv2.imread('./img/'+ file_name + '.png', cv2.IMREAD_COLOR)
    img = img/255
    trimap = cv2.imread('./trimap/'+ file_name + '.png', cv2.IMREAD_GRAYSCALE) # grayscale - two dimension
    
    # prepare masks
    fg_mask = trimap == 255
    bg_mask = trimap == 0
    unknown_mask = True ^ np.logical_or(fg_mask, bg_mask)
    # fill in known fg, bg, and alpha
    alpha = np.zeros(trimap.shape)
    alpha[fg_mask] = 1
    alpha[unknown_mask] = np.nan 
    fg = img*np.repeat(fg_mask[:, :, np.newaxis], 3, axis=2) # expand 2 dimension mask to 3 dimesion(for each R G B channel)
    bg = img*np.repeat(bg_mask[:, :, np.newaxis], 3, axis=2)
    
    
    #########################################
    # TODO: prepare data points and apply GMM
    
    # (Another method) Apply All fg, bg point to train GMM
    # fg_reshape = fg.reshape(fg.shape[0]*fg.shape[1], 3)
    # bg_reshape = bg.reshape(bg.shape[0]*bg.shape[1], 3)
    
    # Apply cv2.dilation to reduce datapoints.  Get the surrounding pixels of the unknown part
    
    #########################################

    # get coordinates of unknown pixels
    unknown = np.argwhere(np.isnan(alpha))
    
    
    # for showing how many pixels were computed
    PASS = 0
    total = len(unknown)
    
    # perpare data used in optimization
    initKernal_size = 25
    iterIncreKernal = 25
    while len(unknown) > 0:
        y,x = unknown[0]
        print("Progress:{}, y = {}, x = {}, img = {}".format(PASS + 1, y, x, img[y][x]))
        kernal_size = initKernal_size
        while True:
            t_alpha = np.zeros(trimap.shape)
            t_alpha[y, x] = 1
            dila_fg_mask, dila_bg_mask = dilationUnknownFgBgNeighbor(t_alpha, kernal_size, fg_mask, bg_mask) # get fg mask, bg mask after dilation process 
            if (np.count_nonzero(dila_fg_mask) >= 30 and np.count_nonzero(dila_bg_mask) >= 30):
                print("kernal_size={}\n".format(kernal_size))
                break
            kernal_size += iterIncreKernal
        dila_fg = img * np.repeat(dila_fg_mask[:, :, np.newaxis], 3, axis=2)
        dila_bg = img * np.repeat(dila_bg_mask[:, :, np.newaxis], 3, axis=2)
        dila_fg = dila_fg.reshape(dila_fg.shape[0]*dila_fg.shape[1], 3)
        dila_bg = dila_bg.reshape(dila_bg.shape[0]*dila_bg.shape[1], 3)
        fg_reduced = dila_fg[~np.all(dila_fg == 0, axis=1)] # reduce datapoints not the surrounding pixels of the unknown part 
        bg_reduced = dila_bg[~np.all(dila_bg == 0, axis=1)] 
            
        # apply GMM to fg and bg points
        num_fg_components = 5
        num_bg_components = 5
        fg_gmm = GaussianMixture(n_components=num_fg_components).fit(fg_reduced)
        bg_gmm = GaussianMixture(n_components=num_bg_components).fit(bg_reduced)
        
        inv_cov_fgmm = np.linalg.inv(fg_gmm.covariances_)
        inv_cov_bgmm = np.linalg.inv(bg_gmm.covariances_)
        mean_fgmm = fg_gmm.means_
        mean_bgmm = bg_gmm.means_
        #print("cov of the fg GMM: \n {}".format(fg_gmm.covariances_))
        #print("cov of the bg GMM: \n {}".format(bg_gmm.covariances_))
        #print("inv c0ov of the fg GMM: \n {}".format(inv_cov_fgmm))
        #print("inv cov of the bg GMM: \n {}".format(inv_cov_bgmm))
        #print("mean of the fg GMM: \n {}".format(mean_fgmm))
        #print("mean of the bg GMM: \n {}".format(mean_bgmm))
    
        ##########################################################################
        # TODO: try to set different initial of alpha and optimize F, B, and alpha
        
        C = img[y][x]
        sigmaC = 0.02
        initAlpha = 0.01
        iterNum = 30
        invSigmaSqr = 1 / (sigmaC**2)
        I = np.eye(3)
        maxlike_pair = np.zeros(2)
        max_F = np.zeros(3)
        max_B = np.zeros(3)
        max_a = np.zeros(0)
        maxlike = -np.inf
        minlike = 1e-6
        
        maxL_3 = np.zeros(3)
        # for each pair of fg and bg
        for i in range(num_fg_components):
            mu_Fi = mean_fgmm[i]
            invSigma_Fi = inv_cov_fgmm[i]
            for j in range(num_bg_components):
                mu_Bj = mean_bgmm[j]
                invSigma_Bj = inv_cov_bgmm[j]
                # set param
                F = []
                B = []
                a = initAlpha
                lastlike = -np.inf # record the last likehood value
                for it in range(iterNum):
                    # fix a, update F and B
                    A11 = invSigma_Fi + I * (a**2) * invSigmaSqr
                    A12 = I * a * (1 - a) * invSigmaSqr
                    A22 = invSigma_Bj + I * ((1 - a) ** 2) *  invSigmaSqr
                    A = np.vstack( (np.hstack((A11, A12)), np.hstack((A12, A22))) )
                    b1 = invSigma_Fi @ mu_Fi + C * a * invSigmaSqr
                    b2 = invSigma_Bj @ mu_Bj + C * (1 - a) * invSigmaSqr
                    b = np.concatenate((b1, b2))
                    X = np.linalg.solve(A, b)
                    #print(X)
                    
                    F = np.maximum(0, np.minimum(1, X[0:3]))
                    B = np.maximum(0, np.minimum(1, X[3:6]))
                    
                    # fix F and B, update a
                    a = np.maximum(0, np.minimum(1, np.dot((C - B), (F - B)) / (np.linalg.norm(F - B)**2)))                
                    #print("F:{}\n B:{} \na:{}\n".format(F, B, a))
                    
                    L_C = -((np.linalg.norm(C - a * F - (1 - a) * B)**2) * invSigmaSqr)
                    #L_F = -((np.dot(np.dot(F - mu_Fi, invSigma_Fi), F - mu_Fi) / 2))
                    L_F = (- ((F - mu_Fi) @ invSigma_Fi @ (F - mu_Fi)) / 2)
                    # L_B = -((np.dot(np.dot(B - mu_Bj, invSigma_Bj), B - mu_Bj) / 2))
                    L_B = (- ((B - mu_Bj) @ invSigma_Bj @ (B - mu_Bj)) / 2)
                    like = L_C + L_F + L_B
                    #print("L_C={}\nL_F={}\nL_B={}\nlike={}".format(L_C, L_F, L_B, like))
                    if maxlike < like:
                        maxlike = like
                        maxlike_pair = [i, j] # record i, j index
                        max_F = F
                        max_B = B
                        max_a = a
                        maxL_3 = [L_C, L_F, L_B]
                    # likehood threshold to stop
                    if abs(like - lastlike) <= minlike:
                        break;
                        
                    lastlike = like;
        # Final F, B, a
        #print('max_L_3={}\nmax_like={}\n'.format(maxL_3, maxProb))
        #print("Index:[{}, {}], maxProb = {}, MAX_F = {}\nMAX_B = {}\na = {}".format(maxlike_pair[0], maxlike_pair[1], maxlike, max_F, max_B, max_a))
        
        alpha[y][x] = max_a
        fg[y][x] = max_F
        bg[y][x] = max_B
        
        
                
        ##########################################################################
        
        unknown = np.delete(unknown, 0, 0)      
        PASS += 1
        sys.stdout.write("\rprogress:\t{}/{}\n".format(PASS,total))
        sys.stdout.flush()
    #print(fg)
    #print(bg)
    
    target_scene = cv2.imread('landscape.png',cv2.IMREAD_COLOR)
    target_scene = cv2.resize(target_scene, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_CUBIC)/255
    
    ####################################################
    # TODO: attach the result foreground to target scene
    
    alpha_extend = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    composite_img = 255 * (alpha_extend * fg + (1 - alpha_extend) * target_scene)
    #print(composite_img)
    grayscale_imgName = 'final/Gray_composite_sigmaC={}_a={}_iterNum={}_Fcomp={}_Bcomp={}_kernal={}_iterIncreKernal={}_fileName={}.png'.format(sigmaC, initAlpha, iterNum, num_fg_components, num_bg_components, initKernal_size, iterIncreKernal, file_name)
    cv2.imwrite(grayscale_imgName, alpha * 255)
    img_name = 'final/composite_sigmaC={}_a={}_iterNum={}_Fcomp={}_Bcomp={}_kernal={}_iterIncreKernal={}_fileName={}.png'.format(sigmaC, initAlpha, iterNum, num_fg_components, num_bg_components, initKernal_size, iterIncreKernal, file_name)
    cv2.imwrite(img_name, composite_img)
    ####################################################
    
