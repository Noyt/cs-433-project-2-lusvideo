# LUS CS433 Project
## Additional libraries
- OpenCV : `pip install opencv-python`
- Torch : `pip install torch torchvision`
- Numpy : `pip install numpy`
- Pandas : `pip install pandas`
- PIL : `pip install Pillow`
- Matplotlib : `pip install matplotlib`

## Dataset
As the data used in this project is sensitive, one should contact Mary-Anne Hartley at mary-anne.hartley@epfl.ch to optain access to the Ultrason butterflynetwork dataset, which was used for this project.

## Repository Structure
Data should be organised in data folder in the following way :
- LUS_main-ipynb
- data/
    - Ultrason_butterflynetwork/

Please note that there needs to be an underscore linking the two words in the folder's name for interpretation purposes.

When running the main notebook `LUS_main.ipynb` additional subfolders are automatically generated under the `data` folder. These contain the datasets generated throughout the project.

An additional `models` folder contains all the models that we have trained. Eventhough this folder is created and filled during the execution of the notebook, we provide it and its content for time saving purposes.

Similarly, a `figures` folder is automatically generated and contains a graphical representation of the models' training. Just like the `models` folder, we are providing it in this repo.

The `Sobel_test` notebook contains an exploratory attempt to using image edge, but was abandonned.

## Covid dataset
At the end of the project, we performed some testing using another dataset and the DeepChest model. The dataset can be optained from Mariko Makhmutova at mariko.makhmutova@epfl.ch, while the model is the property of the lab, therefore an access needs to be granted through the same person.