### translucent principled bunny
Hyper parameters: iteration count 200, others default
Optimization parameters: eta clamped to 1.55, others default

Reference:
```
<bsdf type="principled" id="object_bsdf">
    <rgb name="base_color" value="0.412, 0.824, 0.999"/>
    <float name="roughness" value="0.01"/>
    <float name="spec_trans" value="0.9"/>
    <float name="eta" value="1.49"/>
</bsdf>
```

Optimized:
```
"object_bsdf.eta": [
        0.8558140397071838
    ],
"object_bsdf.roughness.value": [
    0.0010000000474974513
],
"object_bsdf.base_color.value": "[[0.41211390495300293, 0.8714005947113037, 0.9990000128746033]]",
"object_bsdf.spec_trans.value": [
    0.9138074517250061
],
```

#### Conclusion
Yikes! This time roughness looks better, however eta seems to be waay of from the target. At our first approach eta moved away from the target (the value increased), but this time it passed by the reference value (1.49) and move even lower to the value of 0.85. The good news is the other parameters look "better" and fortunately we have other tools at our disposal to handle the "eta" issue. One might argue, since the eta value moved past by the reference value, the learning rate might have a played a role in it. The learning rate for eta seems to be too fast for the optimization. Next, we modify the learning rate of eta from 0.03 to 0.003 and hope the results will look better.