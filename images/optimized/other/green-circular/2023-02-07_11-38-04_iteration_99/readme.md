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
        1.4369902610778809
    ],
    "mat-material-bsdf.roughness.value": [
        0.8628891110420227
    ],
    "mat-material-bsdf.base_color.value": "[[0.027709687128663063, 0.15646228194236755, 0.06963536888360977]]",
    "mat-material-bsdf.spec_trans.value": [
        0.2576342225074768
    ],
```

#### Conclusion
Our initial result do not particularly look good. Since we do not know the reference values, our discussion will be quite speculative. However, one can possibly argue that the roughness value is relatively high. Additionally, although the color value seemed to move in the right direction, the rough texture of the material is missing. Consequently, for our next attempt, we change the representation of color to a bitmap texture and, for the roughness value, set the maximum clamp value of 0.7. The results are shown in Figure X and Table X.