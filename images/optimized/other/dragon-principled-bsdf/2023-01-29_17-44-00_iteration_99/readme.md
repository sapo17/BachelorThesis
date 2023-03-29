### translucent principled bunny
Hyper parameters: everythin default except spp = 8
Optimization parameters: everything default except we use dual buffer method

Reference:
```
<bsdf type="principled" id="object_bsdf">
    <texture name="base_color" type="bitmap">
        <string name="filename" value="gradient.jpg"/> resolution 7200x4800!
    </texture>
    <float name="roughness" value="0.001"/>
    <float name="spec_trans" value="1.0"/>
    <float name="eta" value="1.49"/>
</bsdf>
```

Optimized last iteration:
```
"mat-ObjectBsdf.eta": [
    1.8079876899719238
],
"mat-ObjectBsdf.roughness.value": [
    0.1264638602733612
],
"mat-ObjectBsdf.spec_trans.value": [
    0.0010000000474974513
],
basecolor: see outputted texture file
```

#### Conclusion
Visually, our intial attemp looks fine enough. However, when we take a glance to the optimized parameters, we observe that the values are not near enough to the reference values. This is specifically the case for spec_trans parameter, which seems like it moved completely in the wrong direction. Another interesting parameter is the base color. As mentioned previously, this time, we are using not an RGB color, rather a bitmap texture. Later, we will discuss the bitmap texture that represents the base color parameter in more detail, but first we will focus on the task at our hand. We also observe that the eta parameter moved in the wrong direction. In the previous example, where we estimated Principled BSDF parameters of the bunny, we had similar issue, and we took an approach where we clamped the maximum value of eta. This time, we follow a similar approach for the specular transmission parameter, where we change the minimum clamp value from 0.001 to 0.1. For the moment we don't change the maximum clamp value of eta, and hope that the right movement of spec_trans parameter will affect the movement of eta during optimization.


<!-- Below, move to later -->

The resulting bitmap texture look far away from the reference bitmap texture. The good thing is regarding the optimized bitmap texture is the look in the final rendered image. The look of the optimized bitmap texture is expected, since the optimization begins completely from a completely empty (black) bitmap texture and only those pixel in the bitmap texture are optimized where the mapping of texture takes place. Inevitably, the optimized bitmap texture takes the shape of the object that it is mapped on. One limitation of this approach is that the optimized bitmap texture only works for those camera poses (i.e. only loaded reference images) that our scene description contains. The idea is much easier to grasp visually. In Figure X we show another render of the object with the optimized values, however, this time we rotate the object of interest. Compared to it reference counter part, we observe that the optmized version is clearly incorrect. 

Consequently, one could make the argument, if we use multiple camera poses, we would be also able to optimized the other part of the objects bitmap texture. Fortunately, since material-optimizer provides this functionality, we can test this assumption. In Figure X we show the results where we use multiple reference images for the optimization. Unfortunately, our assumption is invalid. We speculate and direct the reason behind this phenomena to the following: Since the base color parameter is mapped to only one bitmap texture object, we can only optimized one bitmap texture object. Therefore every time the optimization uses another camera position, it tries to optimize the same bitmap texture, and every time it fails to do so, since at every camera pose it needs to *restart* the optimization. As mentioned, this is more or less an educated guess from this authors part. We leave this as future point of analysis.

As in the previous examples, we make use of the tools that in our disposal.
