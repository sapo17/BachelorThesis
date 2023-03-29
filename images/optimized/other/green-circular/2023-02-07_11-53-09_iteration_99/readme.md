### green algea
Hyper parameters: everything default except spp = 8 
Optimization parameters: everything default except the dual buffer method

Reference:
```
?
```


Optimized last it:
```
    "mat-material-bsdf.eta": [
        1.2232520580291748
    ],
    "mat-material-bsdf.roughness.value": [
        0.699999988079071
    ],
    "mat-material-bsdf.spec_trans.value": [
        0.29452475905418396
    ],
    also bitmap texture
```

#### Conclusion
This time our results look much better. The use of bitmap texture definitely added much detail that was missing. Interestingly, the roughness value remained around its maximum clamp value, which might be close to the reference value since the object of interest is not fully transparent. Another parameter we might need to improve is the spec_trans parameter, which might be currently quite low. Consequently, we set the minimum clamp value of spec_trans to 0.4 and restart the optimization. The results are shown in Figure X and Table X.