# Google Landmark Recognition 2019 
## Team *Partials, Callbacks and Hooks*

### Project related links:

* [Team Project Board](https://github.com/users/omaerz/projects/1)




### Tips: 

Neither OpenOffice/LibreOffice nor Google Sheets seem to be able to handle the large csv files. So you can either split them: For example to upload them to Google Sheets split them with `split -d -l 200000 --additional-suffix=.csv train.csv train.part.` and `split -d -l 100000 --additional-suffix=.csv train_attribution.csv train_attribution.part`.
Or use an application that supports large csv files like https://www.csvexplorer.com/ (web based) or http://openrefine.org/ (java app). 



#### How to mount a GCS bucket on an VM instance (running Ubuntu):
First install gcsfuse:
```bash
# install gcs
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update && sudo apt install gcsfuse -y
```
Then make the directory where you want mount the bucket:
```bash
mkdir bucket
```
Login to Google account:
```bash 
gcloud auth application-default login
```
Finally mount the bucket (the GCS bucket's name is *landmark-traing*)
```bash
gcsfuse landmark-training bucket/
```
