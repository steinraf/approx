
# Image Approximation Techniques

## Quadtree

The quadtree algorithm recursively partitions the image into subregions until the local error falls below some threshold or a maximum refinement is reached. Each subregion approximates the image by a constant color.

### Example

Two different subdivision methods are implemented. 

The regular quadtree algorithm splits into 4 equally sized children by default, and 2 if one dimension is below a threshold.

![Shrek regular quadtree](examples/shrek_regular_tree.png)

The adaptive variant does not always use the centroid for splitting but optimizes the choice of the pivot over multiple possible locations.

![Shrek adaptive quadtree](examples/shrek_adaptive_tree.png)


Reference

![Shrek reference](examples/shrek.png)

Result Regular

![Shrek approximated](examples/shrek_regular.png)

Result adaptive

![Shrek approximated](examples/shrek_adaptive.png)


#### Results based on decreasing cell size

![Regular Quadtree Video](examples/refinement_regular.gif)
![Adaptive Quadtree Video](examples/refinement_adaptive.gif)