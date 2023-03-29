### translucent principled bunny
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
    1.4758665561676025
],
"mat-ObjectBsdf.roughness.value": [
    0.00842286180704832
],
"mat-ObjectBsdf.spec_trans.value": [
    0.9542004466056824
],
basecolor: see outputted texture file
```

#### Conclusion
Fortunately, this time both visual and analytical results look good. The optimization ended at iteration 111, since the loss at iteration 111 (0.000981) was lower than the default minimum loss (0.001). Both eta and spec_trans parameters seemed to be quite near to the reference values. To see whether we can get even closer to the reference values, we lower the minimum default error from 0.001 to 0.0001 and start the optimization again with the same configuration.