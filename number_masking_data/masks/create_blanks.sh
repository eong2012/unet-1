for file in *.png
do
  echo "processing $file"
  convert $file -fx 0 -define png:color-type=2 $file
done
