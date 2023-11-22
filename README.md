# Figure-recognition

## Project name: 
* CNN figure properties recognition

![image](https://github.com/Da5ker/Figure-recognition/assets/113497168/6d3ebe24-6483-4c3c-9f94-3a55b6922448)

### Creators: 
* Danil Trotsenko, Aleksei Redkov
  
### Reposetory's description:
- **files** - папка для хранения файлов
    - image_array.npz - zip file with images
    - image_data.csv - dataset with figure properties
    - model.pkl
    - optimizer.pkl
- **images** - folder for images
- **scripts** - folder for script
    - generate_dataset.ipynb - generation of figure images
    - recognition.ipynb - main scripts for recognition
    - preprocess.py - scale and transform data
    - dataset.py - custom dataset
    - model.py - CNN
    - train.py - train scripts

### Setup:

```
conda create --name figurenv --file requirements.txt
conda activate figurenv
```
