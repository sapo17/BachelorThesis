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
        1.4806865453720093
    ],
    "object_bsdf.alpha.value": [
        0.01528790220618248
    ],
    "medium1.albedo.value.value": "[[0.473238468170166, 0.847354531288147, 0.9990000128746033]]",
    "medium1.sigma_t.value.value": [
        0.5024071931838989
    ]
```

Optimized it 69:
```
    "object_bsdf.eta": [
        1.4739885330200195
    ],
    "object_bsdf.alpha.value": [
        0.0010000000474974513
    ],
    "medium1.albedo.value.value": "[[0.5004003047943115, 0.82240891456604, 0.9990000128746033]]",
    "medium1.sigma_t.value.value": [
        0.46556732058525085
    ]
```

#### Conclusion
In our fourth attempt, the optimization ran fully and ended with overall loss rate of 0.000636. This time not only visual results look fine but also the analytical results. We observe that the eta parameter got quite close to its corresponding reference value. The extinction coefficient (sigma\_t) parameter seems to be more far than our previous attempt, but since the visual results are fine enough, we take it as an acceptable result. However, with the tools at our disposal, one could further tune the optmization parameters and hyperparameters to acquire possibily even more exact results. When we switch to scene state where mininmum loss was achieved throughout the optmization, we observe similarly acceptable analytical results. The mininmum loss rate seems (0.000524) to be achieved at iteration 69. We observe that the sigma\_t parameter is closer to its corresponding reference value, however the eta and base color parameters are just a bit more farther compared the last optimization. Since in both cases visual results look fine enough, we believe one could take both as an acceptable result.

