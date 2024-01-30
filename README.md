# Generación y modificación de datos: Un enfoque para la generación de imágenes a partir de diseños textiles

## Descripción del proyecto
Este repositorio alberga el trabajo final para la asignatura de Machine Learning Engineering, un proyecto desarrollado por Pablo Reina Jiménez (github.com/preinaj) y María Lourdes Linares Barrera (github.com/lourdesLB) en el marco del Máster Cloud, Datos y Gestión TI de la Universidad de Sevilla. El objetivo principal del proyecto es explorar la generación de imágenes a partir de diseños textiles, utilizando cGANs. 


## Contenido completo

Debido a las limitaciones de almacenamiento de GitHub, el contenido completo del proyecto, incluyendo código, datasets, generaciones del modelo, documentación y presentación, se encuentra disponible en el siguiente enlace de OneDrive [1]: https://uses0-my.sharepoint.com/personal/marlinbar_alum_us_es/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmarlinbar%5Falum%5Fus%5Fes%2FDocuments%2Fproyecto%2Dsketches%2DMLE&ga=1

Dataset en Kaggle [2]: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset 



## Estructura del repositorio

El proyecto se estructura en torno a los siguientes bloques:  

1. Carpeta `/data`: Incluye imágenes y sketches para entrenamiento y evaluación del modelo. Debido a las limitaciones de subida de ficheros de Github, el contenido esta carpeta no se encuentra en este repositorio pero puede acceder a él en el enlace de OneDrive [1]. Contiene las subcarpetas:
    * `/data/images_all`: Carpeta que debe contener el dataset de imágenes completo de Kaggle.
    * `/data/images_filtered`: Carpeta con el suconjunto de imágenes filtrado por nosotros.
    * `/data/images`: Carpeta con el subconjunto de imágenes filtado, reducido en calidad y con homogeneización de tamaño. Es decir, las imágenes definitivas, divididas en train-test, listas para entrenar/evaluar el modelo.
    * `/data/sketches`: Carpeta con el subconjunto sketches generados con OpenCV (sketches que van a usarse, divididos en train-test), divididos en train-test, listos para entrenar/evaluar el modelo.
    * `/data/sketches2`:  Carpeta con el subconjunto sketches generados con PhotoSketch (sketches que van a usarse, divididos en train-test), divididos en train-test, listos para entrenar/evaluar el modelo.


2. Notebook `generate_dataset.ipynb`: Preprocesa las imágenes para el entrenamiento del modelo, incluyendo filtrado, redimensionamiento, homogeneización y división en conjuntos de entrenamiento y prueba. Este preprocesamiento da como resultados la carpeta `/data` antes descrita, salvo por el hecho de que necesitará descargar las imágenes del dataset de Kaggle sobre la carpeta `/data/images_all` y porque los sketches de PhotoSketch deberá generarlos con la herramienta PhotoSketch [2] y subirlos a la carpeta `/data/sketches2`. Alternativamente puede recurrir al enlace de OneDrive facilitado anteriormente, donde puede descargarse la carpeta `/data` con todos los pares imágenes-sketches ya preprocesados por nosotros para poder entrenar y evaluar los modelos a partir de este punto.


3. Notebook `generate_gan_OpenCV.ipynb`:  Este fichero contiene el código para entrenar la cGAN sobre los pares de imágenes-bocetos de OpenCV, incluyendo resultados o representaciones de generaciones del modelo durante su entrenamiento y generaciones tras el entrenamiento sobre datos de test que el modelo nunca ha visto. El notebook  `generate_gan_PhotoSketch.ipynb` realiza la misma función pero sobre los pares de imágenes-sketches de PhotoSketch.


4. Carpetas `/logs2` y `/logs3`:  que contiene los logs y salidas de los modelos entrenados sobr los pares imágenes-sketches de PhotoSketch o imágenes-sketches de OpenCV (respectivamente). Esta carpeta no ha podido ser cargada en el repositorio de Github por las limitacines de almacenamiento, pero se encuentra subida a la carpeta de OneDrive [1]. En esta carpeta puede encontrar:
    * `/logs{i}/models`: contiene los pesos de los modelos que hemos almacenado después de entrenarlos (`/logs{i}/models/discriminador.h5 o generador.h5`) y los loss del generador obtenidos durante en entrenamiento del modelo a lo largo de las distintas épocas (`/logs{i}/models/train_log.txt`).
    * `/logs{i}/train_images`: generaciones de imágenes para ver cómo evolucionan las imágenes generadas a lo largo del entrenamiento del modelo.
    * `/logs{i}/test_images`: imágenes generadas/salidas del modelo sobre el conjunto de test (imágenes que el modelo no ha visto nunca) una vez ya entrenado.
    

[1] Carpeta de OneDrive con todo el contenido: https://uses0-my.sharepoint.com/:f:/g/personal/marlinbar_alum_us_es/Eo56eNZ_XbVDkNXiG9Wfzj0B1PI-gS8D7dJdvsmv_iy0ZQ?e=kJPvqT

[2] Herramienta PhotoSketch: https://github.com/mtli/PhotoSketch

[3] Dataset de Kaggle: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
