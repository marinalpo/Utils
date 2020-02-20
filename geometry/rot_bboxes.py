import numpy as np

def compute_intersection_by_pairs(polygons, sorted_scores, intersection_th = 0.5):
    from shapely.geometry import Polygon
    from shapely.geometry import asPolygon

    results = []
    for i in range(polygons.shape[0]//2):
        pol_a = asPolygon(polygons[i, :, :])
        pol_b = asPolygon(polygons[i + 1, :, :])
        intersection_area = pol_a.intersection(pol_b)
        iou = intersection_area.area/(pol_a.area+pol_b.area)
        if iou >= intersection_th :
            if(sorted_scores[i] >= sorted_scores[i+1]):
                results.append( [polygons[i,:,:] , sorted_scores[i]] )
            else:
                results.append( [polygons[i+1,:,:] , sorted_scores[i]] )
        else:
            if(sorted_scores[i] >= sorted_scores[i+1]):
                results.append( [polygons[i,:,:] , sorted_scores[i]] )
            else:
                results.append( [polygons[i+1,:,:] , sorted_scores[i]] )
    return results


def filter_bboxes(rboxes, k, c=10):
    """
    rboxes: list, contains: [(np.array(polygon), score),(), ... , ()]
    usage: rboxes = filter_bboxes(rboxes,1, c=10*len(rboxes))
    """
    num_bxs = len(rboxes)
    if(c == 0): # Evita bucle infinit
        return rboxes
    if(num_bxs <= 1):
        return rboxes
    if(num_bxs%2 != 0):
        # We compare by pairs so we need an odd number of boxes
        del rboxes[-1]
        num_bxs = len(rboxes)

    polygons = np.zeros((num_bxs, 4, 2)) # (num_bxs, 8 (points of a polygon, 2 (coordinates of each point)))
    scores = np.zeros(num_bxs)

    for i, (box, score) in enumerate(rboxes):
        box_mat = box.reshape(4,2)
        polygons[i, :, :] = box_mat
        scores[i] = score 
    
    # Compute centroids
    centroids = np.mean(polygons, axis=1)
    indexes_x = np.argsort(centroids[:, 0]) # Ascending order
    indexes_y = np.argsort(centroids[:, 1])
    centroids_x = centroids[indexes_x, 0]
    centroids_y = centroids[indexes_y, 1]
    
    xmax, xmin = centroids_x[-1], centroids_x[0]
    ymax, ymin = centroids_y[-1], centroids_y[0]
    intersection_th = 0.9 # If 2 boxes overlap more than 0.4 the one with max score will remain
    
    if( (xmax - xmin) >= (ymax - ymin) ):
        # Make pairs according to x axis (indexes_x)
        polygons = polygons[indexes_x]
        sorted_scores = scores[indexes_x]
        rboxes = compute_intersection_by_pairs(polygons, sorted_scores)
    else:
        # Make pairs according to y axis (indexes_y)
        polygons = polygons[indexes_y]
        sorted_scores = scores[indexes_y]
        rboxes = compute_intersection_by_pairs(polygons, sorted_scores)
    
    if(len(rboxes) != 1 and len(rboxes) > k):
        c -= 1
        rboxes = filter_bboxes(rboxes, k, c)
    
    return rboxes