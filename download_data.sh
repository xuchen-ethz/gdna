echo Downloading motion sequences...
mkdir data
wget https://scanimate.is.tue.mpg.de/media/upload/demo_data/aist_demo_seq.zip
unzip aist_demo_seq.zip -d ./data/
rm aist_demo_seq.zip
mv ./data/gLO_sBM_cAll_d14_mLO1_ch05 ./data/aist_demo


echo Downloading trained models...
mkdir outputs
wget https://dataset.ait.ethz.ch/downloads/gdna/trained_models.zip
unzip trained_models.zip 
rm trained_models.zip

