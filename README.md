# hamp_processing_python
## Preface
This is repository is used for the post-processing of the HALO microwave package (HAMP) data gained onboard the HALO research aircraft.
In specific, this is a python version that provides the measurements onto a unified grid (in time and vertical) and, from these flight-attitude corrected measurements creates quality controlled data.
## Execution
The processing can be either executed with the python routine 
```python run_hamp_processing.py ``` or, alternatively with interactively showing quicklooks of raw measurements and steps of processed data, using the jupyter notebook ```python run_hamp_processing_notebook.ipnyb ```.  
In ```python run_hamp_processing.main() ```, you define the 
# Processing versions
In ```python run_hamp_processing.main() ``` you define the campaign flights to process and configurate the processing levels that are specified by a version nummer.
Version: 1.0 --> apply nadir perspective: rotation of radar range gates (vertical axis) regarding the given flight attitude onto unified vertical grid. 
Version: 1.x --> additional process steps such as gap filling, outlier removal, side lobe removal, surface mask. All processing steps can be switched off/on, and the version number should be modified correspondingly. (It is planned to predefine the switches by the version number, so that only the number is required as input parameter, so far both have to be set individually).
Version: 2.x --> should be used for the post-calibrated data. This, however, requires additional information (calibration offsets) to be applied to the data. The offsets are derived externally and after the campaign. Version 2.x is thus not feasible for quick performance during the campaign.
