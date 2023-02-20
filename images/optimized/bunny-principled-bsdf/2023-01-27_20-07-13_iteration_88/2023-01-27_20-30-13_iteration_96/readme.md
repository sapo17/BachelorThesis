### translucent principled bunny
Hyper parameters: min. err. 0.0005, spp = 8, others default
Optimization parameters: everything default

Reference:
```
<bsdf type="principled" id="object_bsdf">
    <rgb name="base_color" value="0.412, 0.824, 0.999"/>
    <float name="roughness" value="0.01"/>
    <float name="spec_trans" value="0.9"/>
    <float name="eta" value="1.49"/>
</bsdf>
```

Optimized at iteration #88:
```
"object_bsdf.eta": [
    1.478090763092041
],
"object_bsdf.roughness.value": [
    0.05862249806523323
],
"object_bsdf.base_color.value": "[[0.3877544701099396, 0.7996605634689331, 0.9795681834220886]]",
"object_bsdf.spec_trans.value": [
    0.9205403327941895
],
```

#### Conclusion
Similar to our previous attempt, the dual buffer method introduced by Deng et al. produces either visually or analytically acceptable results and accomplishes this in only 96 iteration. Once again, the optimization procedure stops earlier than the defined iteration count 100, since the loss at step 96 (0.000335) is lower than the modified minimum error value (0.0005).

Consequently, after all these results we showed, we conclude and highly recommend the dual buffer method by Deng et al. for translucent material reconstruction in gradient-based optimization settings. We also conclude that, the default optimization and hyperparameters that the material-optimizer uses is highly compatible with the dual buffer method configuration. Therefore, the translucent material reconstruction results that we are going to present after this point, will always begin with default configuration and the dual buffer method.
