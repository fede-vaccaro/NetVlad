# download paris6k
wget https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz && wget https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
echo "Extracting Paris6K"
tar -xf paris_1.tgz && tar -xf paris_2.tgz
# remove broken paris pics
rm paris/**/paris_moulinrouge_000422.jpg
rm paris/**/paris_pantheon_000960.jpg
rm paris/**/paris_pantheon_000974.jpg
rm paris/**/paris_pantheon_000284.jpg
rm paris/**/paris_museedorsay_001059.jpg
rm paris/**/paris_triomphe_000833.jpg
rm paris/**/paris_triomphe_000662.jpg
rm paris/**/paris_triomphe_000863.jpg
rm paris/**/paris_triomphe_000867.jpg
rm paris/**/paris_pompidou_000467.jpg
rm paris/**/paris_pompidou_000196.jpg
rm paris/**/paris_pompidou_000640.jpg
rm paris/**/paris_pompidou_000201.jpg
rm paris/**/paris_pompidou_000195.jpg
rm paris/**/paris_sacrecoeur_000299.jpg
rm paris/**/paris_sacrecoeur_000353.jpg
rm paris/**/paris_sacrecoeur_000330.jpg
rm paris/**/paris_louvre_000136.jpg
rm paris/**/paris_louvre_000146.jpg
rm paris/**/paris_notredame_000188.jpg
wget https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
mkdir oxford
tar -xf oxbuild_images.tgz -C oxford
mkdir oxford/data
mv *.jpg oxford/data
wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz && wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg2.tar.gz
tar -xf jpg1.tar.gz & tar -xf jpg2.tar.gz
mv jpg holidays
conda install pytorch torchvision h5py pillow=6 tqdm matplotlib pyyaml h5py scikit-learn
