import numpy as np

def connectivity_calc(N, social_class_num, seg_frac, k):
#    if social_class_num >= 2:
#        
#        regular_connection = (1 / social_class_num) #the connectivity for the non-segregated
#        print(regular_connection)
#        segregation = (1-regular_connection) * seg_frac
#        print(segregation)
#        probs = np.zeros((social_class_num, social_class_num))
#        probs[:] = regular_connection - ( segregation / (social_class_num - 1) )
#        np.fill_diagonal(probs, regular_connection + segregation )
#        print(probs)
#
#    
#    elif social_class_num == 1:
#        probs = np.array([[1]])
#    else:
#        print("number of social classes is weird.")
    
    sizes = np.zeros(social_class_num)
    sizes += N/social_class_num #equal sizes for now
    sizes = sizes.astype('int')
    
    anti_ones = np.ones((social_class_num, social_class_num)) - np.identity(social_class_num)
    
    probs = ( ( 1 + (social_class_num - 1) * seg_frac ) / social_class_num ) * np.identity(social_class_num)\
    + ( (1-seg_frac) / social_class_num ) * anti_ones

    print(probs)

    
    predicted_mean_degree = sizes.dot(probs).dot(sizes) / N
    probs = probs / predicted_mean_degree * k #set mean degree to k
    
    
    #print(probs)
    return sizes, probs

