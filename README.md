<a href="http://atvs.ii.uam.es/atvs/">
    <img src="media/BiDA-logo.png" alt="BiDA Lab" title="Bida Lab" align="right" height="150" width="350" target="_blank"/>
</a>

# How Good is ChatGPT at Face Biometrics? A First Look into Recognition, Soft Biometrics, and Explainability

## Table of content

- [Article](#article)
- [Experimental Protocol](#exp_prot)
  - [Databases](#ddbb)
- [Script Usage](#script)
- [References](#references)

## Article

Ivan DeAndres-Tame, Ruben Tolosana, Ruben Vera-Rodriguez, Aythami Morales, Julian Fierrez, Javier Ortega-Garcia, **"How Good is ChatGPT at Face Biometrics? A First Look into Recognition, Soft Biometrics, and Explainability"**, *arXiv preprint arXiv:2401.13641*, doi: [10.48550/arXiv.2401.13641](https://doi.org/10.48550/arXiv.2401.13641), 2023.

<img src="media/G_abstract.jpg" alt="Graphical Abstract" title="Graphical Abstract" align="center" width="90%" target="_blank"/>

## <a name="exp_prot">Experimental Protocol</a>
We provide the experimental protocol used for this research in the *comparisons* folder. It consists of 8 different databases used for face recognition, explainability, and soft-biometrics estimation. For each database, 1000 meaningful comparisons have been selected and evaluated.
### <a name="ddbb">Databases</a>
We analyze the ability of ChatGPT in different application scenarios (i.e., controlled, surveillance, and extreme conditions) and image qualities. For this purpose, we consider the following databases in the evaluation:

* **[Labeled Faces in the Wild (LFW):](http://vis-www.cs.umass.edu/lfw/)** this is a very popular database in the field, containing high-quality images with no hard variations in pose.
* **[QUIS-CAMPI:](http://quiscampi.di.ubi.pt/)** this database comprises videos and images captured in an uncontrolled outdoor setting using a camera positioned approximately 50 meters away from the subjects.
* **[TinyFaces:](https://qmul-tinyface.github.io/)** this database consists of images of extremely low quality, with an average resolution of 20x16 pixels.

In addition to this, we also evaluate the performance of ChatGPT when considering popular challenges in face recognition such as demographic bias, age and pose variations, and occlusions. The following databases are considered in the evaluation, which are also considered in the recent [FRCSyn Challenge](https://frcsyn.github.io/CVPR2024.html):

* **[BUPT-BalancedFace:](http://www.whdeng.cn/RFW/Trainingdataste.html)** this database is specifically designed to tackle performance variations among various ethnic groups. It comprises eight distinct demographic groups formed by a combination of ethnicities (White, Black, Asian, Indian) and gender (Male, Female). 
* **[Celebrities in Frontal-Profile in the Wild (CFP-FP):](http://www.cfpw.io/)** this database presents images from subjects with great changes in pose and different environmental contexts. 
* **[AgeDB:](https://ibug.doc.ic.ac.uk/resources/agedb/)** this database presents diverse images featuring subjects of varying ages in different environmental contexts.
* **[ROF:](https://github.com/ekremerakin/RealWorldOccludedFaces)** this database consists of occluded faces with both upper face occlusion, due to sunglasses, and lower face occlusion, due to masks.

Finally, for the estimation of soft biometrics, we use the **[MAAD-Face](https://github.com/pterhoer/MAAD-Face)** database. This database provides a total of 47 soft-biometric attributes per face image.

## <a name="exp_prot">Scripts Usage</a>
We provide two different scripts to replicate the experiments we performed in our work.
* *combine_img.py* is used to combine the images in the two different image configurations we propose: *1x1 comparisons* and *4x3 comparisons*.
```
python combine_img.py --df_name comparisons/file/.csv 
                      --append_path append/path/for/the/images/in/comparison_file 
                      --path_save_combined where/to/save/1x1images 
                      --path_save_matrix where/to/save/4x3images
```
* *eval_ChatGPR_DDBB.ipynb* is the jupyter notebook used to evaluate the images through ChatGPT. You need to set your OpenAI API Key and the path of the CSV generated used *combine_img.py*.

## <a name="references">References</a>

For further information on the database and on different applications where it has been used, we refer the reader to:

Ivan DeAndres-Tame, Ruben Tolosana, Ruben Vera-Rodriguez, Aythami Morales, Julian Fierrez, Javier Ortega-Garcia, **"How Good is ChatGPT at Face Biometrics? A First Look into Recognition, Soft Biometrics, and Explainability"**, *arXiv preprint arXiv:2401.13641*, doi: [10.48550/arXiv.2401.13641](https://doi.org/10.48550/arXiv.2401.13641), 2023.

Please remember to reference the above articles on any work made public, whatever the form, based directly or indirectly on any part of the article.