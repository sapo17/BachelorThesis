### translucent principled bunny
Multiple reference images
Hyper parameters: everything default except spp = 16
Optimization parameters: everything default except we use dual buffer method, spec_trans: min clamp = 0.11

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
        1.502232313156128
    ],
    "mat-ObjectBsdf.roughness.value": [
        0.0019726522732526064
    ],
    "mat-ObjectBsdf.spec_trans.value": [
        0.9728695154190063
    ],
basecolor: see outputted texture file
```

#### Conclusion
Consequently, one could make the argument, if we use multiple camera poses, we would be also able to optimized the other part of the objects bitmap texture. Fortunately, since material-optimizer provides this functionality, we can test this assumption. In Figure X we show the results where we use multiple reference images for the optimization. Note that we use the same configuration as in our previous successful attempt (i.e. spp=16, iteration count = 200, min. error: 0.001, spec\_trans: min clamp: 0.11, dual buffer method). In total we use 4 sensors; where we place the camera front, back, left and right side of the dragon. The results are shown in Figure X and Table X. Fortunately our assumption was indeed correct, with multiple camera poses, we are able to optimize the bitmap texture of the object more precisely, and this can be also seen in the outputted bitmap texture in Figure X. We also observe another advantage of using multiple camera poses in Table X. Compared to our previous successful attempt where we used only one sensor, we examine that the optimized values are now even more close to their corresponding reference values.