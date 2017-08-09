for file in *.png
do
  echo "processing $file"
  convert $file -channel G -separate -define png:color-type=2 $file
done
