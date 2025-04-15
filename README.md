
# Image Approximation Techniques

## Quadtree

The quadtree algorithm recursively partitions the image into 4 subregions until the local error falls below some threshold. Each subregion approximates the image by a constant color.

![Shrek quadtree](examples/shrek_quadtree.png)

In this way we can reduce the geometric resolution of the image adaptively.

Reference

![Shrek reference](examples/shrek.png)

Result

![Shrek approximated](examples/shrek_approximation.png)