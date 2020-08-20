import numpy as np

def connectivity_calc(N, social_class_num, seg_frac):
    if social_class_num >= 2:
        
        regular_connection = (1 / social_class_num) #the connectivity for the non-segregated
        segregation = regular_connection * seg_frac
        probs = np.zeros((social_class_num, social_class_num))
        probs[:] = regular_connection - ( segregation / (social_class_num - 1) )
        np.fill_diagonal(probs, regular_connection + segregation )
    
    
    elif social_class_num == 1:
        probs = np.array([[1]])
    else:
        print("number of social classes is weird.")
    
    sizes = np.zeros(social_class_num)
    sizes += N/social_class_num #equal sizes for now
    sizes = sizes.astype('int')
    
    predicted_mean_degree = sizes.dot(probs).dot(sizes) / N
    probs = probs / predicted_mean_degree * 6 #set mean degree to 6
    
    return sizes, probs

