# Google Landmark Recognition 2019 
## Team *Partials, Callbacks and Hooks*

### Project related links:

* [Team Project Board](https://github.com/users/omaerz/projects/1)




### Tips: 

Neither OpenOffice/LibreOffice nor Google Sheets seem to be able to handle the large csv files. So you can either split them: For example to upload them to Google Sheets split them with `split -d -l 200000 --additional-suffix=.csv train.csv train.part.` and `split -d -l 100000 --additional-suffix=.csv train_attribution.csv train_attribution.part`.
Or use an application that supports large csv files like https://www.csvexplorer.com/ (web based) or http://openrefine.org/ (java app). 
