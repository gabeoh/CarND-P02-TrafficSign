# CarND-P02-TrafficSign
CarND-P02-TrafficSign implements a neural network model to recognize German
traffic signs.  It trains, validates, and tests the neural network model
using dataset provided in the
[German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

Once the model is trained, validated, and tested, it is tested against a new
set of German traffic sign images found on web.  It further performs error
analyses and data visualizations in order to get better insights into the
neural network behaviors.


## File Structure
### Project Requirements
- **[Traffic_Sign_Classifier.ipynb](Traffic_Sign_Classifier.ipynb)** - Project IPython notebook
- **??** - HTML output of the Ipython notebook
- **[P02_writeup.md](P02_writeup.md)** - Project write-up report

### Additional Files
- **py-src** - Directory containing raw python scripts used in the project
- **image** - Additional traffic sign images found on the web
- **results** - Project outputs such as plots and processed images 
- **[signnames.csv](signnames.csv)** - CSV file containing mappings from
  classId to the sign name

### Not Included
- **data** - Download and extract [Udacity Traffic Signs Data](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip)
  to this directory


## Getting Started
### [Download ZIP](https://github.com/gabeoh/CarND-P02-TrafficSign/archive/master.zip) or Git Clone
```
git clone https://github.com/gabeoh/CarND-P02-TrafficSign.git
```

### Setup environment

You can set up the environment following
[CarND-Term1-Starter-Kit - Miniconda](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md).
This will install following packages required to run this application.

- Miniconda
- Python
- Jupyter Notebook

### Usage

There are two ways of running this project.

#### Jupyter Notebook
Open `Traffic_Sign_Classifier.ipynb`, the project IPython notebook, using Jupyter Notebook.
```
jupyter notebook Traffic_Sign_Classifier.ipynb
```

#### Running Python Scripts
You can also run Python scripts directly from command line.
```
$ cd py-src

# Train, validate, and test the neural network model on the provided dataset
$ python p02_trafficsign_01.py

# Run a prediction on new dataset. Analyze the prediction processes. 
$ python p02_trafficsign_02.py
```

## License
Licensed under [MIT](LICENSE) License.

