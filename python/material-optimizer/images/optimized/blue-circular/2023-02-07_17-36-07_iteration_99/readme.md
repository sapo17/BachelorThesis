### blue algea
Hyper parameters: everything default except spp = 8 
Optimization parameters: everything default except the dual buffer method, rougness max clamp=0.6, min. clamp spec_trans=0.5

Reference:
```
?
```


Optimized last it:
```
    "mat-material-bsdf.eta": [
        1.4209004640579224
    ],
    "mat-material-bsdf.roughness.value": [
        0.5884355306625366
    ],
    "mat-material-bsdf.spec_trans.value": [
        0.4000000059604645
    ],
    also bitmap texture
```

#### Conclusion
The results look relatively good. Interestingly, the roughness and spec_trans parameters remained relatively close to the minimum and maximum clamp values. Consequently, we discard the maximum and minimum clamp values defined for the roughness and spec_trans parameter, and restart the optimization. The results are shown in Figure X and Table X.