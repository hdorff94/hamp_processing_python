# hamp_processing_python
## Preface
This is repository is used for the post-processing of the HALO microwave package (HAMP) data gained onboard the HALO research aircraft.
In specific, this is a python version that provides the measurements onto a unified grid (in time and vertical) and, from these flight-attitude corrected measurements creates quality controlled data.
## Raw data
HAMP comprises the measurements from the radiometers in 26 microwave channels shown below for a specific research flight RF04 from:
![Raw_HAMP_Tb_RF04_20220314](https://github.com/hdorff94/hamp_processing_python/assets/39411449/6ebf5a1a-7ed2-4b05-8e13-f5a316dfbbe4)
In addition, it contains the Ka-Band radar, as shown analogously for RF04, below:
![raw_radar_quicklook_20220314](https://github.com/hdorff94/hamp_processing_python/assets/39411449/7ab6b428-9592-4811-90e0-7cc362b3dc12)
## Execution
The processing can be either executed with the python routine 
```python run_hamp_processing.py ``` or, alternatively with interactively showing quicklooks of raw measurements and steps of processed data, using the jupyter notebook ```python run_hamp_processing_notebook.ipnyb ```.  
In ```python run_hamp_processing.main() ```, you execute you can run the processing in the konsole, but be aware that processing steps are configurated in ```main()```. The processing primarily access the processing classes located in 
the folder * [src](https://github.com/hdorff94/hamp_processing_python/tree/main/src)
# Processing versions
In ```python run_hamp_processing.main() ``` you define the campaign flights to process and configurate the processing levels that are specified by a version nummer.
Version: 1.0 --> apply nadir perspective: rotation of radar range gates (vertical axis) regarding the given flight attitude onto unified vertical grid. 
Version: 1.x --> additional process steps such as gap filling, outlier removal, side lobe removal, surface mask. All processing steps can be switched off/on, and the version number should be modified correspondingly. (It is planned to predefine the switches by the version number, so that only the number is required as input parameter, so far both have to be set individually).
Version: 2.x --> should be used for the post-calibrated data. This, however, requires additional information (calibration offsets) to be applied to the data. The offsets are derived externally and after the campaign. Version 2.x is thus not feasible for quick performance during the campaign.
# Processed data
Exemplarly shown for RF04.
## Radar
![unified_radar_dbz_ldr_quicklook_RF04_20220314](https://github.com/hdorff94/hamp_processing_python/assets/39411449/3e2881b2-2e88-4b4f-8e08-dc03acd58750)
## HAMP
![unified_radiometer_tb_quicklook_RF04_20220314](https://github.com/hdorff94/hamp_processing_python/assets/39411449/7a3d4fc4-6eea-494f-9133-f01c98b0fbd0)
