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
        1.624756097793579
    ],
    "object_bsdf.alpha.value": [
        0.003591743530705571
    ],
    "medium1.albedo.value.value": "[[0.4091211259365082, 0.8614102005958557, 0.9990000128746033]]",
    "medium1.sigma_t.value.value": [
        0.4762713313102722
    ]
```

#### Conclusion
In our second attempt the optimization ends at iteration 66, since the overall loss (0.000973) was lower than the default minimum error rate (0.001). Visually the results look good. Analytically almost all optmized values look fine except the eta parameter. Interestingly, although the overall loss in our last iteration was lower than the defined minimum error, the eta parameter again seemed to moved in the wrong direction. Since this configuration provided us with almost successfull results, we extend our configuration where we set the maximum clamp value of eta to 1.55, decrease the mininmum error rate from 0.001 to 0.0001, and lower the iteration count from 200 back to 100, andhope more exact analytical results. The results are shown in Figure X and Table X.