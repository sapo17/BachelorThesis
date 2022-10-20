# LP Softwarepraktikum mit Bachelorarbeit (2022W)

- Name: Saip Can Hasbay
- Student Number: 01428723

## Notes

### 01.10.22

- Added guidelines
- Might be necessary:
  - Contact a professor, for a possible topic.

### 07.10.22

- [Slides](https://bit.ly/3Cm4xX1) available for the 07.10.22
- Bis abmeldefrist: topic necessary 14. Oct
- 4 Milestones:
  1. concrete Milestone identification
  2. 1.5th Month: prototype
  3. 3th Month: final impl. and tested
  4. 31.01: submission thesis + presentation
- Verification:
  - Presentation: 10%
    - Required: > 5%
  - Design & Code: 45%
    - Required: > 22.5%
  - Thesis: 45%
    - Required: > 22.5%
- Submission:
  - Thesis: LaTeX [Template available](https://zid.univie.ac.at/overleaf)
  - Moodle:
    - Thesis
    - Tool
  - GitLab
    - Ask to supervisor (**open**)
- Timeline:
  - 14 Oct: Topic and acknowlegment
- Currently everyone is on the waitlist
  - In case of arrangement w/ supervisor, contact Mr. Klas, Student will be officially part of the Thesis Process
- Topic
  - GFX
    - torsten.möller
  - SIP
  - Entertainment comp
    - helmut.hlavacs
  - Neuroinformatics
    - wentrup: 12.10, 16:45-20.00

### 13.10.22

- See email traffic w/ Prof. Hlavacs, for another thesis topic possiblity (**not relevant**)
- See email traffic w/ Prof. Moeller, David Hahn for thesis arrangements
  - Send email for a meeting w/ David and Torsten regarding clarifiying thesis timeline(**done**)
  - Send email to Prof. Klas for clarifying thesis topic (**done**)
- Thesis Topic: A modified/formulated version of [Shape optimization of light sources with global illumination objectives](https://www.cg.tuwien.ac.at/courses/projekte/Shape-optimization-light-sources-global-illumination-objectives)
- First notes from the talk w/ David:
  - TODO:
    - Analyse:
      - emissive texture
    - Read Related work:
      - [LightGuider: Guiding Interactive Lighting Design using Suggestions, Provenance, and Quality Visualization](https://doi.org/10.1109/TVCG.2019.2934658) (**done**)
      - [Procedural Design of Exterior Lighting for Buildings with Complex Constraints](https://doi.org/10.1145/2629573)
      - [Narrow-Band Topology Optimization on a Sparsely Populated Grid](https://doi.org/10.1145/3272127.3275012)
    - [Material parameter estimation](https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Zeltner2021MonteCarlo.pdf) (**done**)
      - Try out in [Mitsuba](https://github.com/mitsuba-renderer/mitsuba3)
      - Use mitsuba and set up experimental situation, like box and material in the middle
        - give material parameters so we can achieve original image from real life
    - [Nerf](https://www.matthewtancik.com/nerf)
      - encode radiance field Nerfs w/ neurol networks
      - how much data does one neeeds for a neuiral structure
      - make a comparision w/ different renderers
    - half page or full page: abstract and when you want to finish

### 14.10.22

- Done following mitsuba tutorials:
  - [Mitsuba 3: Youtube tutorials](https://www.youtube.com/watch?v=9Ja9buZx0Cs)
  - [Mitsuba 3: Read the tocs tutorial, until Cuastics Opimization](https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/caustics_optimization.html)
- References:
  - Diederik P Kingma and Jimmy Ba. Adam: a method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
  - Stochastic Gradient Descent
    - Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and momentum in deep learning. In International conference on machine learning, 1139?1147. 2013

### 15.10.22

- Prof Moeller, David (see also emails from 14.10):

  - Torsten

    > great. yes, let us meet. I am on conference next week, however. May I
    > suggest to find a time the week after? E.g. Mon, Oct 24th 12:30-3pm would
    > be possible for me. May I kindly ask you, Saip-Can, to create some slide
    > (ppt or keynote or beamer or whatever you like) to detail:
    >
    > - the problem/research question
    > - approach
    > - milestones

  - David

    > Sure, 24th afternoon works for me. I like the idea of having a summary slide.
    > Saip-Can, maybe you can summarize what we discussed yesterday (and also which direction seems most interesting to you), and let me know in case any further questions come up until then.

  - Can
    - Arrange meeting for 24.10.22, 14:00-15:00 (**done**)

- Read:
  - [Material parameter estimation](https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Zeltner2021MonteCarlo.pdf) (**done**)
    - notes about the paper (**done**, see docs)
- drjit quick start tutorial (**done**)
- Work on _Material parameter estimation_ until at the end of 18.10.22

### 16.10.22

- Ideas:
  - As mentioned on the future work section of the paper _material parameter estimation_, one could add NEE that targets differentiates gradients specifically
    - Implementation of NEE
    - Comparision of NEE vs other methodologies
  - Inverse rendering, scene reconstruction from real world images:
    1. Real world image
    2. Inverse rendering (using mitsuba 3)
    3. Inverse modelling (mitsuba scene params to CAD app, e.g.)
    4. Compare results, user test (?)

### 17/18.10.22

- Worked on optimizing multiple scene parameters, see _optimization-test.ipynb_
