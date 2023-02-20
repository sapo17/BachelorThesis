### translucent principled bunny
Hyper parameters: everythin default
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
        1.5110257863998413
    ],
    "object_bsdf.roughness.value": [
        0.0010000000474974513
    ],
    "object_bsdf.base_color.value": "[[0.4294069707393646, 0.798065185546875, 0.9483137726783752]]",
    "object_bsdf.spec_trans.value": [
        0.8364441394805908
    ],
```

#### Conclusion
In fascinating turn of events, such as Ray Allen's amazing game-tying 3-pointer in Game 6 (NBA Finals 2013), or when Carl Sagan instructed the NASA team of Voyager 1 for a photograph of planet Earth from a record distance of about 6 billion kilometers, as part of that day's family portrait series of images of the solar system where the Earth's apparent size is less than a pixel and the planet appears as a tiny dot against the vastness of space among bands of sunglight reflected by the camera (TODO cite book pale blue dot and wiki), the dual buffer method steals the show by reconstructing the bunny-objects translucent material properties in only 88 iterations. Although we set the iteration count to 100, the iteration stops at iteration 88 because the default minimum error value for the loss defined by material-optimizer is 0.001, and since the loss rate at iteration 88 is 0.000837 the iteration stops. The resulting image looks quite similar to the reference image and translucent parameter values are also quite near to the target. But since we are quite ambitious -- and naive -- we give it another shot, where we set the minimum error value for the loss from 0.001 to 0.0005 and increase the spp to 8. 
