### translucent principled bunny
Hyper parameters: iteration count 200, spp = 8
Optimization parameters: eta clamped to 1.55, eta lr = 0.008

Reference:
```
<bsdf type="principled" id="object_bsdf">
    <rgb name="base_color" value="0.412, 0.824, 0.999"/>
    <float name="roughness" value="0.01"/>
    <float name="spec_trans" value="0.9"/>
    <float name="eta" value="1.49"/>
</bsdf>
```

Optimized last iteration:
```
"object_bsdf.eta": [
    1.2197390794754028
],
"object_bsdf.roughness.value": [
    0.00521653750911355
],
"object_bsdf.base_color.value": "[[0.447121798992157, 0.8523852825164795, 0.9990000128746033]]",
"object_bsdf.spec_trans.value": [
    0.8411227464675903
],
```

Optimized at minimum error #139:
```
"object_bsdf.eta": [
        1.4848335981369019
],
"object_bsdf.roughness.value": [
    0.030202288180589676
],
"object_bsdf.base_color.value": "[[0.41312655806541443, 0.8624976873397827, 0.9990000128746033]]",
"object_bsdf.spec_trans.value": [
    0.9245197772979736
],
```

#### Conclusion
This time, at the end of the 199 iteration, the result does not particularly look good. Specifically, the eta parameter -- again -- seems to be quite low. However, one thing that we can immediately note is the "smoothnes" of the parameter error plot. In our previous attemps, the parameter error plot look almost always quite "wobbly". We can clearly see in the parameter error plot before the 175 iteration the loss seemed to be lower than after the 175 iteration, thus the latest iteration 199. This is actually good news, since our tool provides a "magic-button" that allows the user to switch between the minimum loss and last iteration of the optimized scene state. Consequently, the "magic-button" comes to our rescue and we switch to the scene state with the minimum error rate and voila -- visually, the optimized image is indeed looking much better! We output the optimized parameters at iteration number 139 and observe very promising estimated parameter values. The parameter error space at the end of iteration 139 is the following:

xxx

To understand whether this result was achieved by modifying the samples per pixel value (thus the noise introduced by MC sampling with the default spp value 4) or learning rate, we take the same configuration as in the previous example and this time change only the learning rate from 0.003 to 0.008 (i.e. we dont touch the spp value).
