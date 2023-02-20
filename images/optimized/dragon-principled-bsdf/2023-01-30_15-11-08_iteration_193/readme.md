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
    1.545840859413147
],
"mat-ObjectBsdf.roughness.value": [
    0.011919979006052017
],
"mat-ObjectBsdf.spec_trans.value": [
    0.9770709872245789
],
basecolor: see outputted texture file
```

#### Conclusion
This time the optimization finishes in iteration 200. At the end of the iteration overall loss rate is 0.000523. When we switch to the optimization state with the minimum loss throughout the whole procedure, we observe minimum loss at iteration 193 (For conciseness we discuss only the optimization state at iteration 193, since analytical results are quite identical to the optimization state at iteration 200). Interestingly, although the overall loss rate is lower than our previous attemps (i.e. previous overall loss rate was 0.000981), and visually both optimized images at iteration 111 and 193 look quite identical, we observe that the eta parameter is actually farther from the target. In conclusion, we should remind ourselves, that lower error rate not necessarily refers to better material estimation.


Fortunately, this time both visual and analytical results look good. The optimization ended at iteration 111, since the loss at iteration 111 (0.000981) was lower than the default minimum loss (0.001). Both eta and spec_trans parameters seemed to be quite near to the reference values. To see whether we can get even closer to the reference values, we lower the minimum default error from 0.001 to 0.0001 and start the optimization again with the same configuration.