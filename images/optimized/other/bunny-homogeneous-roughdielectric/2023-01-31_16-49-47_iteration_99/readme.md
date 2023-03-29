### translucent principled bunny
Hyper parameters: everything default except spp = 8 
Optimization parameters: everything default except the dual buffer method

Reference:
```
<bsdf type="roughdielectric" id="object_bsdf">
    <string name="distribution" value="ggx"/>
    <float name="alpha" value="0.01"/>
    <string name="int_ior" value="acrylic glass"/>
    <string name="ext_ior" value="air"/>
</bsdf>
<medium id="medium1" type="homogeneous">
    <float name="scale" value="5"/>
    <!-- <ref id="hg_phase1" name="phase_hg"/> -->
    <phase type="isotropic" />
    <float name="sigma_t" value="0.4"/>
    <rgb name="albedo" value="0.412, 0.824, 0.999"/>
    <boolean name="has_spectral_extinction" value="false"/>
    <boolean name="sample_emitters" value="false"/>
</medium>
```


Optimized last it:
```
 "object_bsdf.eta": [
        1.9670724868774414
    ],
    "object_bsdf.alpha.value": [
        0.0010000000474974513
    ],
    "medium1.albedo.value.value": "[[0.5902319550514221, 0.928102970123291, 0.9990000128746033]]",
    "medium1.sigma_t.value.value": [
        0.6037914752960205
    ]
```

#### Conclusion
Similar to our first example a bunny object is shown in Figure X. Since we obtained quite successfull results in our previous attemps, we use the dual buffer method by Deng et al. and samples per pixel (spp) value of 8. The results from our first attemp is shown in Figure X and Table X. 

Although visually our result look fine enough, surprisingly our initial attempt provides us analytically incorrect results. This is especially the case for the eta parameter. Once again, we observe that the eta parameter is moved in the wrong direction. Since we had a similar issue previously, we change our configuration where we increase the spp value from 8 to 16 and iteration count from 100 to 200, and hope that the eta parameter will converge into the right direction and consequently affect other parameters as well. The results from our second attempt is shown in Figure X and Table X.