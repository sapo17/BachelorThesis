### green algea
Hyper parameters: everything default except spp = 8 
Optimization parameters: everything default except the dual buffer method, rougness max clamp=0.7, min. clamp spec_trans=0.4

Reference:
```
?
```


Optimized last it:
```
    "mat-material-bsdf.eta": [
        1.222919225692749
    ],
    "mat-material-bsdf.roughness.value": [
        0.699999988079071
    ],
    "mat-material-bsdf.spec_trans.value": [
        0.44359827041625977
    ],
    also bitmap texture
```

#### Conclusion
Visually, the results look similar to our previous attempt. The only significant analytical change to our previous attempt is the spec_trans parameter, which the additional constraint we set most likely help to increase compared to our previous attempt.