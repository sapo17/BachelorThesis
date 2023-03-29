### translucent principled bunny
Hyper parameters: iteration count 200, others default
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
    1.4248024225234985
],
"object_bsdf.roughness.value": [
    0.0010000000474974513
],
"object_bsdf.base_color.value": "[[0.43605485558509827, 0.9318716526031494, 0.9990000128746033]]",
"object_bsdf.spec_trans.value": [
    0.9464256763458252
],
```

Optimized at minimum error #69:
```
"object_bsdf.eta": [
    1.5387115478515625
],
"object_bsdf.roughness.value": [
    0.014547315426170826
],
"object_bsdf.base_color.value": "[[0.3243766725063324, 0.5904024839401245, 0.6991850733757019]]",
"object_bsdf.spec_trans.value": [
    0.8784424066543579
],
```

#### Conclusion
The visual results are not particularly "bad", however we immediately notice the parameter error plot is quite wobbly. Even the use of our "magic-button" does not help us, the values are not waay off, but definitely not quite there yet, especially as in our previous succesfull attempt. Consequently, we conclude that our assumption regarding the noise introduced by MC sampling indeed justifiable, and it does indeed effect the optimization procedure. Subsequently, we get ahead of ourselves, and try the same configuration as in our successfull attempt, however this time with samples per pixel value 16 (i.e. spp = 16, eta lr = 0.008, eta max. clamp value = 1.55, iteration count = 200).
